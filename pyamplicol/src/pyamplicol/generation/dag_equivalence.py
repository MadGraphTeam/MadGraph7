# SPDX-License-Identifier: 0BSD
"""Proof-gated recursive current-value reuse for generated DAGs.

Current identity includes colour-sector and ordering metadata needed to build
and reduce amplitudes.  Those fields do not necessarily change the numerical
current value.  This module proves such value equivalences from the complete
recursive computation instead of guessing them from particle names or PDGs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum, isfinite
from typing import Literal, TypeAlias

from ..models.base import Model, VertexEvaluationEquivalence
from .contracts import runtime_coupling_parameter_names
from .dag_types import AmplitudeRoot, CurrentNode, GenericDAG, InteractionNode

_ComplexWeight: TypeAlias = tuple[float, float]
_CurrentContract: TypeAlias = tuple[object, ...]
_EvaluationKey: TypeAlias = tuple[object, ...]
_CurrentTermVector: TypeAlias = tuple[tuple[_EvaluationKey, _ComplexWeight], ...]
_DiscoveryMode: TypeAlias = Literal["diagnostic", "certified-reuse"]

RELATION_DISCOVERY_SCHEMA_VERSION = 1
RELATION_DISCOVERY_CERTIFICATE_ALGORITHM = "exact-binary64-term-vector-replay-v1"
_MAX_REJECTED_DIAGNOSTICS = 16


@dataclass(frozen=True, slots=True)
class _CurrentValueEquivalence:
    """Exact relation ``current = factor * representative``."""

    representative_id: int
    factor: _ComplexWeight


@dataclass(frozen=True, slots=True)
class _CanonicalProjectiveTermVector:
    """A deterministic lookup key and its concrete normalization factor."""

    term_vector: _CurrentTermVector
    factor: _ComplexWeight


@dataclass(frozen=True, slots=True)
class _ProjectiveExpressionRepresentative:
    """One concrete current retained as a projective-class representative."""

    representative_id: int
    term_vector: _CurrentTermVector
    normalization_factor: _ComplexWeight


@dataclass(frozen=True, slots=True)
class DynamicColorProjectionCertificate:
    """Fail-closed proof summary for one multilinear color projection.

    The projection removes only ``CurrentIndex.color_state.sector_id``.
    Accuracy, line groups, basis keys, every non-colour current field, and the
    complete downstream physical-selector domain remain part of the class
    key.  A projected kernel or closure row is emitted only when the old
    ordered parent tuples are exactly the full Cartesian product of the
    corresponding member classes, with no duplicate tuple.
    """

    abi: str
    source_revision: str | None
    source_semantics_sha256: str
    selector_domains_sha256: str
    current_class_members_sha256: str
    old_to_new_current_ids: tuple[int, ...]
    current_remap_sha256: str
    interaction_groups_sha256: str
    closure_groups_sha256: str
    rectangle_cardinalities_sha256: str
    row_identity_sha256: str
    equality_check_status: str
    retained_color_metadata: str
    root_sector_policy: str
    before_current_count: int
    after_current_count: int
    before_interaction_count: int
    after_interaction_count: int
    before_evaluation_count: int
    after_evaluation_count: int
    before_amplitude_root_count: int
    after_amplitude_root_count: int
    projected_current_class_count: int
    rectangular_interaction_group_count: int
    rectangular_closure_group_count: int
    split_current_class_count: int

    @property
    def applied(self) -> bool:
        return (
            self.after_current_count < self.before_current_count
            or self.after_interaction_count < self.before_interaction_count
            or self.after_amplitude_root_count < self.before_amplitude_root_count
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "source_revision": self.source_revision,
            "source_semantics_sha256": self.source_semantics_sha256,
            "selector_domains_sha256": self.selector_domains_sha256,
            "current_class_members_sha256": self.current_class_members_sha256,
            "old_to_new_current_ids": list(self.old_to_new_current_ids),
            "current_remap_sha256": self.current_remap_sha256,
            "interaction_groups_sha256": self.interaction_groups_sha256,
            "closure_groups_sha256": self.closure_groups_sha256,
            "rectangle_cardinalities_sha256": (
                self.rectangle_cardinalities_sha256
            ),
            "row_identity_sha256": self.row_identity_sha256,
            "equality_check_status": self.equality_check_status,
            "retained_color_metadata": self.retained_color_metadata,
            "root_sector_policy": self.root_sector_policy,
            "applied": self.applied,
            "before_current_count": self.before_current_count,
            "after_current_count": self.after_current_count,
            "before_interaction_count": self.before_interaction_count,
            "after_interaction_count": self.after_interaction_count,
            "before_evaluation_count": self.before_evaluation_count,
            "after_evaluation_count": self.after_evaluation_count,
            "before_amplitude_root_count": self.before_amplitude_root_count,
            "after_amplitude_root_count": self.after_amplitude_root_count,
            "projected_current_class_count": self.projected_current_class_count,
            "rectangular_interaction_group_count": (
                self.rectangular_interaction_group_count
            ),
            "rectangular_closure_group_count": self.rectangular_closure_group_count,
            "split_current_class_count": self.split_current_class_count,
        }


@dataclass(frozen=True, slots=True)
class ExactCurrentRelationCertificate:
    """Replayable proof that one concrete current is an exact multiple."""

    current_id: int
    representative_id: int
    factor: _ComplexWeight
    current_term_vector_sha256: str
    representative_term_vector_sha256: str
    proof_sha256: str
    algorithm: str = RELATION_DISCOVERY_CERTIFICATE_ALGORITHM

    def to_json_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "current_id": self.current_id,
            "representative_id": self.representative_id,
            "factor_binary64": [self.factor[0].hex(), self.factor[1].hex()],
            "current_term_vector_sha256": self.current_term_vector_sha256,
            "representative_term_vector_sha256": (
                self.representative_term_vector_sha256
            ),
            "proof_sha256": self.proof_sha256,
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> ExactCurrentRelationCertificate:
        """Decode the exact representation emitted into generation metadata."""

        expected_fields = {
            "algorithm",
            "current_id",
            "representative_id",
            "factor_binary64",
            "current_term_vector_sha256",
            "representative_term_vector_sha256",
            "proof_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("relation certificate fields are not canonical")
        try:
            algorithm = payload["algorithm"]
            current_id = payload["current_id"]
            representative_id = payload["representative_id"]
            raw_factor = payload["factor_binary64"]
            current_digest = payload["current_term_vector_sha256"]
            representative_digest = payload["representative_term_vector_sha256"]
            proof_digest = payload["proof_sha256"]
        except KeyError as error:
            raise ValueError(
                f"relation certificate is missing {error.args[0]!r}"
            ) from error
        if algorithm != RELATION_DISCOVERY_CERTIFICATE_ALGORITHM:
            raise ValueError("relation certificate uses an unsupported algorithm")
        if (
            type(current_id) is not int
            or type(representative_id) is not int
            or current_id < 0
            or representative_id < 0
        ):
            raise ValueError("relation certificate current IDs must be nonnegative")
        if (
            not isinstance(raw_factor, list)
            or len(raw_factor) != 2
            or any(not isinstance(component, str) for component in raw_factor)
        ):
            raise ValueError(
                "relation certificate factor must contain two binary64 hex strings"
            )
        try:
            factor = (float.fromhex(raw_factor[0]), float.fromhex(raw_factor[1]))
        except ValueError as error:
            raise ValueError(
                "relation certificate factor has invalid binary64 encoding"
            ) from error
        if not _complex_weight_is_finite(factor) or factor == (0.0, 0.0):
            raise ValueError("relation certificate factor must be finite and nonzero")
        digests = (current_digest, representative_digest, proof_digest)
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ValueError("relation certificate digests must be lowercase SHA-256")
        assert isinstance(current_digest, str)
        assert isinstance(representative_digest, str)
        assert isinstance(proof_digest, str)
        certificate = cls(
            current_id=current_id,
            representative_id=representative_id,
            factor=factor,
            current_term_vector_sha256=current_digest,
            representative_term_vector_sha256=representative_digest,
            proof_sha256=proof_digest,
        )
        if (
            _certificate_proof_sha256(
                current_id=certificate.current_id,
                representative_id=certificate.representative_id,
                factor=certificate.factor,
                current_term_vector_sha256=(certificate.current_term_vector_sha256),
                representative_term_vector_sha256=(
                    certificate.representative_term_vector_sha256
                ),
            )
            != certificate.proof_sha256
        ):
            raise ValueError("relation certificate proof digest does not replay")
        return certificate


@dataclass(frozen=True, slots=True)
class RelationDiscoveryReport:
    """Mode- and colour-scoped outcome of one opt-in discovery pass."""

    requested_mode: str
    state: str
    execution_mode: str
    color_accuracy: str
    representation: str
    precision_digits: int
    probe_count: int
    seed: int
    probe_status: str
    certificate_replay_status: str
    numerical_candidate_count: int
    uncertified_candidate_count: int
    exact_certified_relation_count: int
    applied_relation_count: int
    interaction_evaluation_count_before: int | None
    interaction_evaluation_count_after: int | None
    certificates: tuple[ExactCurrentRelationCertificate, ...] = ()
    rejected_candidates: tuple[dict[str, object], ...] = ()
    follow_up_boundary: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": RELATION_DISCOVERY_SCHEMA_VERSION,
            "requested_mode": self.requested_mode,
            "state": self.state,
            "scope": {
                "execution_mode": self.execution_mode,
                "color_accuracy": self.color_accuracy,
                "representation": self.representation,
            },
            "probe": {
                "status": self.probe_status,
                "precision_digits": self.precision_digits,
                "probe_count": self.probe_count,
                "seed": self.seed,
                "deterministic": True,
                "candidate_only": True,
            },
            "certificate_replay": {
                "algorithm": RELATION_DISCOVERY_CERTIFICATE_ALGORITHM,
                "status": self.certificate_replay_status,
            },
            "numerical_candidate_count": self.numerical_candidate_count,
            "uncertified_candidate_count": self.uncertified_candidate_count,
            "exact_certified_relation_count": (self.exact_certified_relation_count),
            "applied_relation_count": self.applied_relation_count,
            "interaction_evaluation_count_before": (
                self.interaction_evaluation_count_before
            ),
            "interaction_evaluation_count_after": (
                self.interaction_evaluation_count_after
            ),
            "interaction_evaluation_savings": (
                None
                if self.interaction_evaluation_count_before is None
                or self.interaction_evaluation_count_after is None
                else max(
                    0,
                    self.interaction_evaluation_count_before
                    - self.interaction_evaluation_count_after,
                )
            ),
            "certificates": [
                certificate.to_json_dict() for certificate in self.certificates
            ],
            "rejected_candidates": list(self.rejected_candidates),
            "follow_up_boundary": self.follow_up_boundary,
        }


@dataclass(frozen=True, slots=True)
class RelationDiscoveryResult:
    dag: GenericDAG
    report: RelationDiscoveryReport


class _NumericalRelationDiscovery:
    """Nominate projective pairs numerically, then require an exact replay."""

    def __init__(
        self,
        *,
        precision_digits: int,
        probe_count: int,
        seed: int,
    ) -> None:
        if precision_digits < 80:
            raise ValueError("relation discovery precision must be at least 80 digits")
        if probe_count < 2:
            raise ValueError("relation discovery requires at least two probe points")
        if seed < 0:
            raise ValueError("relation discovery seed must be nonnegative")
        self.precision_digits = int(precision_digits)
        self.probe_count = int(probe_count)
        self.seed = int(seed)
        self.numerical_candidate_count = 0
        self.uncertified_candidate_count = 0
        self.certificates: list[ExactCurrentRelationCertificate] = []
        self.rejected_candidates: list[dict[str, object]] = []
        self._representatives_by_fingerprint: dict[
            tuple[_CurrentContract, tuple[object, ...]],
            list[tuple[int, _CurrentTermVector]],
        ] = {}

    def consider(
        self,
        *,
        current_id: int,
        contract: _CurrentContract,
        term_vector: _CurrentTermVector,
        baseline: _CurrentValueEquivalence,
    ) -> _CurrentValueEquivalence:
        fingerprint = _numerical_projective_fingerprint(
            term_vector,
            precision_digits=self.precision_digits,
            probe_count=self.probe_count,
            seed=self.seed,
        )
        if fingerprint is None:
            return baseline
        key = (contract, fingerprint)
        representatives = self._representatives_by_fingerprint.setdefault(key, [])

        promoted = baseline
        if baseline.representative_id == current_id and baseline.factor == (1.0, 0.0):
            exact_matches: list[tuple[int, _ComplexWeight, _CurrentTermVector]] = []
            for representative_id, representative_vector in representatives:
                self.numerical_candidate_count += 1
                factor = _certify_exact_term_vector_relation(
                    representative_vector,
                    term_vector,
                )
                if factor is None:
                    self.uncertified_candidate_count += 1
                    if len(self.rejected_candidates) < _MAX_REJECTED_DIAGNOSTICS:
                        self.rejected_candidates.append(
                            {
                                "current_id": current_id,
                                "representative_id": representative_id,
                                "reason": "numerical-candidate-lacks-exact-certificate",
                            }
                        )
                    continue
                exact_matches.append((representative_id, factor, representative_vector))
            if exact_matches:
                representative_id, factor, representative_vector = min(
                    exact_matches,
                    key=lambda item: item[0],
                )
                certificate = _build_exact_current_relation_certificate(
                    current_id=current_id,
                    representative_id=representative_id,
                    factor=factor,
                    current_term_vector=term_vector,
                    representative_term_vector=representative_vector,
                )
                self.certificates.append(certificate)
                promoted = _CurrentValueEquivalence(
                    representative_id=representative_id,
                    factor=factor,
                )

        representatives.append((current_id, term_vector))
        return promoted


class RecursiveEvaluationReuseTracker:
    """Certify recursive-current reuse as external subsets are completed."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self._kernel_equivalences: dict[int, VertexEvaluationEquivalence] = {}
        self._runtime_coupling_identities: dict[
            tuple[int, tuple[int, int, int], tuple[float, float]],
            tuple[tuple[float, float], tuple[tuple[int, str], ...]],
        ] = {}
        self._current_equivalences: list[_CurrentValueEquivalence] = []
        self._source_representative_by_key: dict[tuple[object, ...], int] = {}
        self._equivalence_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
        ] = {}
        self._projective_representatives_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector],
            list[_ProjectiveExpressionRepresentative],
        ] = {}
        self._evaluation_group_by_key: dict[_EvaluationKey, int] = {}
        self._coefficients_by_result: list[
            dict[int, _ComplexWeight | list[_ComplexWeight]] | None
        ] = []

    def register_source(self, current: CurrentNode) -> None:
        if not current.is_source:
            raise ValueError("recursive-reuse source registration requires a source")
        if current.source_leg_label is None or current.source_helicity is None:
            raise ValueError(
                f"source current {current.id} lacks physical source metadata"
            )
        contract = _current_evaluation_contract(current)
        source_key = (
            contract,
            int(current.source_leg_label),
            int(current.source_helicity),
        )
        representative_id = self._source_representative_by_key.setdefault(
            source_key,
            current.id,
        )
        self._append_current_equivalence(
            current,
            _CurrentValueEquivalence(representative_id, (1.0, 0.0)),
        )

    def interaction_evaluation(
        self,
        *,
        vertex_kind: int,
        vertex_particles: tuple[int, int, int],
        left_id: int,
        right_id: int,
        result: CurrentNode,
        coupling: tuple[float, float],
        color_weight: _ComplexWeight,
    ) -> tuple[int, _ComplexWeight]:
        kernel_equivalence = _kernel_equivalence(
            self._model,
            vertex_kind,
            self._kernel_equivalences,
        )
        try:
            left = self._current_equivalences[left_id]
            right = self._current_equivalences[right_id]
        except IndexError as error:
            raise ValueError(
                "online recursive reuse requires completed parent subsets"
            ) from error
        canonical_inputs, evaluation_factor = _canonical_interaction_evaluation(
            kernel_equivalence,
            left_id=left_id,
            right_id=right_id,
            left=left,
            right=right,
        )
        coupling_key = (vertex_kind, vertex_particles, coupling)
        coupling_identity = self._runtime_coupling_identities.get(coupling_key)
        if coupling_identity is None:
            coupling_identity = _runtime_coupling_identity(
                self._model,
                vertex_kind=vertex_kind,
                vertex_particles=vertex_particles,
                coupling=coupling,
            )
            self._runtime_coupling_identities[coupling_key] = coupling_identity
        evaluation_key = (
            kernel_equivalence.class_id,
            canonical_inputs,
            int(result.index.particle_id),
            int(result.index.chirality),
            coupling_identity,
        )
        evaluation_group_id = self._evaluation_group_by_key.setdefault(
            evaluation_key,
            len(self._evaluation_group_by_key),
        )
        if result.id >= len(self._coefficients_by_result):
            self._coefficients_by_result.extend(
                [None] * (result.id + 1 - len(self._coefficients_by_result))
            )
        coefficients_by_group = self._coefficients_by_result[result.id]
        if coefficients_by_group is None:
            coefficients_by_group = {}
            self._coefficients_by_result[result.id] = coefficients_by_group
        coefficient = _complex_weight_mul(color_weight, evaluation_factor)
        coefficients = coefficients_by_group.get(evaluation_group_id)
        if coefficients is None:
            coefficients_by_group[evaluation_group_id] = coefficient
        elif isinstance(coefficients, list):
            coefficients.append(coefficient)
        else:
            coefficients_by_group[evaluation_group_id] = [coefficients, coefficient]
        return evaluation_group_id, evaluation_factor

    def finalize_currents(
        self,
        currents: Iterable[CurrentNode],
    ) -> None:
        for current in currents:
            if current.is_source:
                raise ValueError("generated-current finalization received a source")
            coefficients_by_group = self._coefficients_by_result[current.id]
            self._coefficients_by_result[current.id] = None
            terms: list[tuple[_EvaluationKey, _ComplexWeight]] = []
            for group_id in sorted(coefficients_by_group or ()):
                assert coefficients_by_group is not None
                coefficients = coefficients_by_group[group_id]
                if isinstance(coefficients, list):
                    coefficient = (
                        _canonical_zero(fsum(value[0] for value in coefficients)),
                        _canonical_zero(fsum(value[1] for value in coefficients)),
                    )
                else:
                    coefficient = _canonical_complex_weight(coefficients)
                if coefficient != (0.0, 0.0):
                    terms.append(((group_id,), coefficient))
            term_vector = tuple(terms)
            contract = _current_evaluation_contract(current)
            equivalence = _classify_current_term_vector(
                current_id=current.id,
                contract=contract,
                term_vector=term_vector,
                equivalence_by_expression=self._equivalence_by_expression,
                projective_representatives_by_expression=(
                    self._projective_representatives_by_expression
                ),
            )
            self._append_current_equivalence(
                current,
                equivalence,
            )

    def _append_current_equivalence(
        self,
        current: CurrentNode,
        equivalence: _CurrentValueEquivalence,
    ) -> None:
        if current.id != len(self._current_equivalences):
            raise ValueError(
                "online recursive reuse requires currents in contiguous ID order"
            )
        self._current_equivalences.append(equivalence)
        if current.id >= len(self._coefficients_by_result):
            self._coefficients_by_result.append(None)


def _canonical_kernel_evaluation(
    equivalence: VertexEvaluationEquivalence,
    left_id: int,
    right_id: int,
) -> tuple[tuple[int, int], _ComplexWeight]:
    """Return canonical representative inputs and the concrete-kernel factor."""

    canonical_inputs = (left_id, right_id)
    if equivalence.input_order == (1, 0):
        canonical_inputs = (right_id, left_id)
    factor = equivalence.factor
    if (
        equivalence.input_exchange_factor is not None
        and canonical_inputs[1] < canonical_inputs[0]
    ):
        canonical_inputs = (canonical_inputs[1], canonical_inputs[0])
        factor = _complex_weight_mul(
            factor,
            equivalence.input_exchange_factor,
        )
    return canonical_inputs, factor


def _canonical_interaction_evaluation(
    equivalence: VertexEvaluationEquivalence,
    *,
    left_id: int,
    right_id: int,
    left: _CurrentValueEquivalence,
    right: _CurrentValueEquivalence,
) -> tuple[tuple[int, int], _ComplexWeight]:
    """Compose recursive factors exactly or retain the concrete inputs.

    Projective equivalence is an exact statement about the ideal algebraic
    current.  Its recursive factor may be moved through another kernel only
    when both complex products remain finite and exactly representable as
    binary64.  Otherwise this interaction fails closed to its concrete parent
    IDs while retaining the pre-existing model-certified kernel symmetry.
    """

    canonical_inputs, kernel_factor = _canonical_kernel_evaluation(
        equivalence,
        left.representative_id,
        right.representative_id,
    )
    parent_factor = _exact_representable_complex_product(left.factor, right.factor)
    if parent_factor is not None:
        evaluation_factor = _exact_representable_complex_product(
            kernel_factor,
            parent_factor,
        )
        if evaluation_factor is not None:
            return canonical_inputs, evaluation_factor
    return _canonical_kernel_evaluation(equivalence, left_id, right_id)


def assign_recursive_current_evaluation_reuse(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Share kernel evaluations through exactly proven current equivalences.

    The proof is recursive.  Duplicate source wavefunctions form the base
    classes.  A generated current joins an existing class only when its full
    vector of model-certified kernel terms and coefficients is byte-exactly
    equal, opposite, or algebraically projective through a finite nonzero
    binary64 factor whose coefficient reconstruction is bit-exact.  Moving a
    general projective factor changes floating-point association, so runtime
    parity remains a tolerance-checked numerical contract rather than a claim
    of bit-identical materialized currents.  Recursive factor products fail
    closed unless they are finite and exactly representable.  The current
    contract keeps every field consumed by source, kernel, and propagator
    evaluation; colour bookkeeping, ordering metadata, and ancestry bit
    allocation are deliberately excluded. Ordering may differ only through
    the exact model-certified input permutation and reflection factors
    included in the term signature below.

    This recovers AmpliCol-style reflection fan-out, but also recognizes exact
    reuse across colour sectors and helicity subgraphs.  No approximate
    numerical comparison or process/model-family classification is involved.
    """

    if not dag.interactions:
        return dag

    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
    current_equivalences = _derive_current_value_equivalences(
        dag,
        model,
        equivalence_by_kind=equivalence_by_kind,
    )
    return _rewrite_interaction_evaluation_reuse(
        dag,
        model,
        current_equivalences=current_equivalences,
        equivalence_by_kind=equivalence_by_kind,
    )


def discover_recursive_evaluation_relations(
    dag: GenericDAG,
    model: Model,
    *,
    mode: _DiscoveryMode,
    execution_mode: Literal["compiled", "eager"],
    precision_digits: int = 96,
    probe_count: int = 4,
    seed: int = 0x5059414D,
) -> RelationDiscoveryResult:
    """Discover candidates numerically and promote only exact DAG relations.

    Probe values are deterministic high-precision synthetic evaluations of the
    full current term vectors. They are candidate indexes only. Every promoted
    relation is independently replayed over the complete binary64 coefficient
    vectors, and unsafe recursive factor composition continues to fail closed.
    """

    if mode not in {"diagnostic", "certified-reuse"}:
        raise ValueError(f"unsupported relation discovery mode {mode!r}")
    if execution_mode not in {"compiled", "eager"}:
        raise ValueError(
            "GenericDAG relation discovery supports compiled and eager execution"
        )

    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    discovery = _NumericalRelationDiscovery(
        precision_digits=precision_digits,
        probe_count=probe_count,
        seed=seed,
    )
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
    current_equivalences = _derive_current_value_equivalences(
        dag,
        model,
        equivalence_by_kind=equivalence_by_kind,
        discovery=discovery,
    )
    promoted = _rewrite_interaction_evaluation_reuse(
        dag,
        model,
        current_equivalences=current_equivalences,
        equivalence_by_kind=equivalence_by_kind,
    )
    if discovery.certificates and not verify_dag_relation_certificates(
        dag,
        model,
        discovery.certificates,
    ):
        discovery.uncertified_candidate_count += len(discovery.certificates)
        if len(discovery.rejected_candidates) < _MAX_REJECTED_DIAGNOSTICS:
            discovery.rejected_candidates.append(
                {
                    "reason": "exact-certificate-chain-replay-failed",
                    "relation_count": len(discovery.certificates),
                }
            )
        discovery.certificates.clear()
        promoted = baseline
    interaction_changed = _interaction_reuse_signature(
        promoted
    ) != _interaction_reuse_signature(baseline)
    apply_promotions = (
        mode == "certified-reuse"
        and bool(discovery.certificates)
        and interaction_changed
    )
    result_dag = promoted if apply_promotions else dag
    report = RelationDiscoveryReport(
        requested_mode=mode,
        state=("exact-certified-applied" if apply_promotions else "diagnostic-only"),
        execution_mode=execution_mode,
        color_accuracy=str(dag.process.color_accuracy),
        representation="generic-dag",
        precision_digits=precision_digits,
        probe_count=probe_count,
        seed=seed,
        probe_status="completed",
        certificate_replay_status=(
            "verified" if discovery.certificates else "no-certified-relations"
        ),
        numerical_candidate_count=discovery.numerical_candidate_count,
        uncertified_candidate_count=discovery.uncertified_candidate_count,
        exact_certified_relation_count=len(discovery.certificates),
        applied_relation_count=(len(discovery.certificates) if apply_promotions else 0),
        interaction_evaluation_count_before=baseline.interaction_evaluation_count,
        interaction_evaluation_count_after=(
            promoted.interaction_evaluation_count
            if apply_promotions or mode == "diagnostic"
            else baseline.interaction_evaluation_count
        ),
        certificates=tuple(discovery.certificates),
        rejected_candidates=tuple(discovery.rejected_candidates),
        follow_up_boundary=(
            "direct vertex-kernel equivalence remains model-certificate-owned; "
            "this pass promotes exact current proportionality and its induced "
            "interaction fan-out only"
        ),
    )
    return RelationDiscoveryResult(dag=result_dag, report=report)


def recurrence_relation_discovery_diagnostic(
    *,
    requested_mode: _DiscoveryMode,
    color_accuracy: Literal["lc", "nlc", "full"],
    precision_digits: int,
    probe_count: int,
    seed: int,
) -> RelationDiscoveryReport:
    """Return the shared manifest shape without mutating a recurrence schedule."""

    return RelationDiscoveryReport(
        requested_mode=requested_mode,
        state="diagnostic-only",
        execution_mode="recurrence",
        color_accuracy=color_accuracy,
        representation="recurrence-schedule",
        precision_digits=precision_digits,
        probe_count=probe_count,
        seed=seed,
        probe_status="not-run",
        certificate_replay_status="unavailable",
        numerical_candidate_count=0,
        uncertified_candidate_count=0,
        exact_certified_relation_count=0,
        applied_relation_count=0,
        interaction_evaluation_count_before=None,
        interaction_evaluation_count_after=None,
        follow_up_boundary=(
            "recurrence promotion requires an exact schedule-level replay "
            "certificate over the lowered Rust current/contribution tables"
        ),
    )


def _rewrite_interaction_evaluation_reuse(
    dag: GenericDAG,
    model: Model,
    *,
    current_equivalences: tuple[_CurrentValueEquivalence, ...],
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence],
) -> GenericDAG:
    evaluation_group_by_key: dict[_EvaluationKey, int] = {}
    interactions: list[InteractionNode] = []

    for interaction in dag.interactions:
        kernel_equivalence = _kernel_equivalence(
            model,
            interaction.vertex_kind,
            equivalence_by_kind,
        )
        left = current_equivalences[interaction.left_id]
        right = current_equivalences[interaction.right_id]
        canonical_inputs, evaluation_factor = _canonical_interaction_evaluation(
            kernel_equivalence,
            left_id=interaction.left_id,
            right_id=interaction.right_id,
            left=left,
            right=right,
        )
        result = dag.currents[interaction.result_id]
        evaluation_key = (
            kernel_equivalence.class_id,
            canonical_inputs,
            int(result.index.particle_id),
            int(result.index.chirality),
            _runtime_coupling_identity(
                model,
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                coupling=interaction.coupling,
            ),
        )
        evaluation_group_id = evaluation_group_by_key.setdefault(
            evaluation_key,
            len(evaluation_group_by_key),
        )
        interactions.append(
            replace(
                interaction,
                evaluation_group_id=evaluation_group_id,
                evaluation_factor=evaluation_factor,
            )
        )

    rewritten = tuple(interactions)
    if rewritten == dag.interactions:
        return dag
    return replace(dag, interactions=rewritten)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_root_color_sector_id(dag: GenericDAG, root: AmplitudeRoot) -> int:
    """Resolve the physical sector before any representative is selected."""

    if root.color_sector_id is not None:
        return int(root.color_sector_id)
    return int(dag.currents[root.left_id].index.color_state.sector_id)


def _downstream_color_selector_domains(
    dag: GenericDAG,
) -> tuple[tuple[int, ...], ...]:
    """Return the complete physical LC selector domain of every current."""

    domains: list[set[int]] = [set() for _current in dag.currents]
    for root in dag.amplitude_roots:
        sector_id = _resolved_root_color_sector_id(dag, root)
        domains[root.left_id].add(sector_id)
        domains[root.right_id].add(sector_id)
    for interaction in reversed(dag.interactions):
        result_domains = domains[interaction.result_id]
        if not result_domains:
            continue
        domains[interaction.left_id].update(result_domains)
        domains[interaction.right_id].update(result_domains)
    return tuple(tuple(sorted(values)) for values in domains)


def _dynamic_color_projection_key(
    current: CurrentNode,
    selector_domains: tuple[int, ...],
) -> tuple[object, ...]:
    """Return exact current identity with only dynamic LC sector omitted.

    ``basis_key`` participates in coherent amplitude grouping and eager
    metadata, while ``line_groups`` is the exact open-colour identity.  Both
    therefore remain invariant within a projected class.  The lowerings use
    particle/chirality/momentum plus the explicit row identity for numerical
    kernels; the retained representative sector is metadata only after every
    amplitude root has received its pre-projection resolved sector.
    """

    index = current.index
    return (
        int(index.particle_id),
        int(index.external_mask),
        index.external_labels,
        index.ordered_external_labels,
        int(index.helicity_ancestry),
        int(index.chirality),
        index.spin_state,
        index.flavour_flow,
        index.quantum_number_flow,
        index.color_state.accuracy,
        index.color_state.line_groups,
        index.color_state.basis_key,
        int(index.momentum_mask),
        index.coupling_orders,
        index.auxiliary_kind,
        int(current.dimension),
        selector_domains,
    )


def _projection_partition(
    dag: GenericDAG,
    selector_domains: tuple[tuple[int, ...], ...],
    split_current_ids: set[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for current in dag.currents:
        # External source rows remain singleton.  Their physical source route
        # is an initialization contract, not a multilinear recurrence row.
        if current.is_source or current.id in split_current_ids:
            key = ("singleton", current.id)
        else:
            key = (
                "dynamic-color",
                _dynamic_color_projection_key(
                    current,
                    selector_domains[current.id],
                ),
            )
        grouped[key].append(current.id)
    classes = tuple(
        sorted(
            (tuple(sorted(members)) for members in grouped.values()),
            key=lambda members: (
                len(dag.currents[members[0]].index.external_labels),
                members[0],
            ),
        )
    )
    class_by_current = [-1] * len(dag.currents)
    for class_id, members in enumerate(classes):
        for current_id in members:
            class_by_current[current_id] = class_id
    if any(class_id < 0 for class_id in class_by_current):
        raise ValueError("dynamic-color projection left a current unpartitioned")
    return classes, tuple(class_by_current)


def _interaction_projection_key(
    interaction: InteractionNode,
    class_by_current: tuple[int, ...],
    *,
    keep_row_distinct: bool,
) -> tuple[object, ...]:
    return (
        class_by_current[interaction.result_id],
        class_by_current[interaction.left_id],
        class_by_current[interaction.right_id],
        int(interaction.vertex_kind),
        interaction.vertex_particles,
        interaction.coupling,
        interaction.color_weight,
        interaction.lowering_backend,
        bool(interaction.full_tensor_network_ready),
        interaction.id if keep_row_distinct else None,
    )


def _root_projection_key(
    dag: GenericDAG,
    root: AmplitudeRoot,
    class_by_current: tuple[int, ...],
    *,
    keep_row_distinct: bool,
) -> tuple[object, ...]:
    return (
        class_by_current[root.left_id],
        class_by_current[root.right_id],
        root.kind,
        root.color_weight,
        root.contraction_ir,
        _resolved_root_color_sector_id(dag, root),
        root.vertex_kind,
        root.vertex_particles,
        root.coupling,
        float(root.helicity_weight),
        root.id if keep_row_distinct else None,
    )


def _projection_groups(
    dag: GenericDAG,
    classes: tuple[tuple[int, ...], ...],
    class_by_current: tuple[int, ...],
) -> tuple[
    dict[tuple[object, ...], list[InteractionNode]],
    dict[tuple[object, ...], list[AmplitudeRoot]],
]:
    interaction_groups: dict[tuple[object, ...], list[InteractionNode]] = defaultdict(
        list
    )
    for interaction in dag.interactions:
        involved = (
            class_by_current[interaction.result_id],
            class_by_current[interaction.left_id],
            class_by_current[interaction.right_id],
        )
        keep_distinct = all(len(classes[class_id]) == 1 for class_id in involved)
        interaction_groups[
            _interaction_projection_key(
                interaction,
                class_by_current,
                keep_row_distinct=keep_distinct,
            )
        ].append(interaction)

    root_groups: dict[tuple[object, ...], list[AmplitudeRoot]] = defaultdict(list)
    for root in dag.amplitude_roots:
        involved = (
            class_by_current[root.left_id],
            class_by_current[root.right_id],
        )
        keep_distinct = all(len(classes[class_id]) == 1 for class_id in involved)
        root_groups[
            _root_projection_key(
                dag,
                root,
                class_by_current,
                keep_row_distinct=keep_distinct,
            )
        ].append(root)
    return interaction_groups, root_groups


def _invalid_projection_classes(
    dag: GenericDAG,
    classes: tuple[tuple[int, ...], ...],
    class_by_current: tuple[int, ...],
    interaction_groups: dict[tuple[object, ...], list[InteractionNode]],
    root_groups: dict[tuple[object, ...], list[AmplitudeRoot]],
) -> set[int]:
    """Return every non-singleton class implicated in a non-rectangle."""

    invalid: set[int] = set()
    produced_by_class: dict[int, set[int]] = defaultdict(set)
    for rows in interaction_groups.values():
        first = rows[0]
        result_class = class_by_current[first.result_id]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        produced_by_class[result_class].update(row.result_id for row in rows)
        if all(
            len(classes[class_id]) == 1
            for class_id in (result_class, left_class, right_class)
        ):
            continue
        actual = [(row.left_id, row.right_id) for row in rows]
        expected = {
            (left_id, right_id)
            for left_id in classes[left_class]
            for right_id in classes[right_class]
        }
        if len(actual) != len(set(actual)) or set(actual) != expected:
            invalid.update(
                class_id
                for class_id in (result_class, left_class, right_class)
                if len(classes[class_id]) > 1
            )

    for class_id, members in enumerate(classes):
        if len(members) == 1 or dag.currents[members[0]].is_source:
            continue
        if produced_by_class.get(class_id, set()) != set(members):
            invalid.add(class_id)

    for roots in root_groups.values():
        first = roots[0]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        if len(classes[left_class]) == len(classes[right_class]) == 1:
            continue
        actual = [(root.left_id, root.right_id) for root in roots]
        expected = {
            (left_id, right_id)
            for left_id in classes[left_class]
            for right_id in classes[right_class]
        }
        if len(actual) != len(set(actual)) or set(actual) != expected:
            invalid.update(
                class_id
                for class_id in (left_class, right_class)
                if len(classes[class_id]) > 1
            )
    return invalid


_DYNAMIC_COLOR_PROJECTION_ABI = (
    "pyamplicol-generic-dag-dynamic-color-projection-v2"
)
_RETAINED_COLOR_METADATA_POLICY = (
    "accuracy-line-groups-basis-key-identical; representative-sector-metadata-only"
)
_ROOT_SECTOR_POLICY = (
    "pre-projection-resolved-sector-is-explicit-and-never-cross-projected"
)


def _initial_dynamic_color_projection_certificate(
    dag: GenericDAG,
    *,
    source_revision: str | None,
    equality_check_status: str,
) -> DynamicColorProjectionCertificate:
    identity = tuple(range(len(dag.currents)))
    return DynamicColorProjectionCertificate(
        abi=_DYNAMIC_COLOR_PROJECTION_ABI,
        source_revision=source_revision,
        source_semantics_sha256=_canonical_sha256(dag.to_json_dict()),
        selector_domains_sha256=_canonical_sha256([]),
        current_class_members_sha256=_canonical_sha256(
            [[current.id] for current in dag.currents]
        ),
        old_to_new_current_ids=identity,
        current_remap_sha256=_canonical_sha256(list(identity)),
        interaction_groups_sha256=_canonical_sha256(
            [[interaction.id] for interaction in dag.interactions]
        ),
        closure_groups_sha256=_canonical_sha256(
            [[root.id] for root in dag.amplitude_roots]
        ),
        rectangle_cardinalities_sha256=_canonical_sha256([]),
        row_identity_sha256=_canonical_sha256(
            {
                "interactions": [
                    interaction.to_json_dict() for interaction in dag.interactions
                ],
                "roots": [root.to_json_dict() for root in dag.amplitude_roots],
            }
        ),
        equality_check_status=equality_check_status,
        retained_color_metadata=_RETAINED_COLOR_METADATA_POLICY,
        root_sector_policy=_ROOT_SECTOR_POLICY,
        before_current_count=len(dag.currents),
        after_current_count=len(dag.currents),
        before_interaction_count=len(dag.interactions),
        after_interaction_count=len(dag.interactions),
        before_evaluation_count=dag.interaction_evaluation_count,
        after_evaluation_count=dag.interaction_evaluation_count,
        before_amplitude_root_count=len(dag.amplitude_roots),
        after_amplitude_root_count=len(dag.amplitude_roots),
        projected_current_class_count=0,
        rectangular_interaction_group_count=0,
        rectangular_closure_group_count=0,
        split_current_class_count=0,
    )


def _projection_proof_payloads(
    dag: GenericDAG,
    selector_domains: tuple[tuple[int, ...], ...],
    classes: tuple[tuple[int, ...], ...],
    class_by_current: tuple[int, ...],
    interaction_groups: dict[tuple[object, ...], list[InteractionNode]],
    root_groups: dict[tuple[object, ...], list[AmplitudeRoot]],
) -> dict[str, object]:
    """Return the complete deterministic witness committed by the certificate."""

    ordered_interaction_groups = sorted(
        interaction_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    )
    ordered_root_groups = sorted(
        root_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    )
    rectangle_cardinalities: list[dict[str, object]] = []
    row_identities: list[dict[str, object]] = []
    for rows in ordered_interaction_groups:
        first = rows[0]
        result_class = class_by_current[first.result_id]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        actual_pairs = sorted((row.left_id, row.right_id) for row in rows)
        rectangle_cardinalities.append(
            {
                "kind": "interaction",
                "row_ids": sorted(row.id for row in rows),
                "result_class_size": len(classes[result_class]),
                "left_class_size": len(classes[left_class]),
                "right_class_size": len(classes[right_class]),
                "row_count": len(rows),
                "unique_parent_pair_count": len(set(actual_pairs)),
                "expected_parent_pair_count": (
                    len(classes[left_class]) * len(classes[right_class])
                ),
            }
        )
        row_identities.append(
            {
                "kind": "interaction",
                "result_class_id": result_class,
                "left_class_id": left_class,
                "right_class_id": right_class,
                "vertex_kind": int(first.vertex_kind),
                "vertex_particles": list(first.vertex_particles),
                "coupling": list(first.coupling),
                "color_weight": list(first.color_weight),
                "lowering_backend": first.lowering_backend,
                "full_tensor_network_ready": bool(
                    first.full_tensor_network_ready
                ),
                "row_ids": sorted(row.id for row in rows),
            }
        )
    for roots in ordered_root_groups:
        first = roots[0]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        actual_pairs = sorted((root.left_id, root.right_id) for root in roots)
        resolved_sector = _resolved_root_color_sector_id(dag, first)
        rectangle_cardinalities.append(
            {
                "kind": "closure",
                "row_ids": sorted(root.id for root in roots),
                "left_class_size": len(classes[left_class]),
                "right_class_size": len(classes[right_class]),
                "row_count": len(roots),
                "unique_parent_pair_count": len(set(actual_pairs)),
                "expected_parent_pair_count": (
                    len(classes[left_class]) * len(classes[right_class])
                ),
                "resolved_color_sector_id": resolved_sector,
            }
        )
        row_identities.append(
            {
                "kind": "closure",
                "left_class_id": left_class,
                "right_class_id": right_class,
                "root_kind": first.kind,
                "color_weight": list(first.color_weight),
                "contraction_ir": first.contraction_ir.to_json_dict(),
                "resolved_color_sector_id": resolved_sector,
                "vertex_kind": first.vertex_kind,
                "vertex_particles": (
                    None
                    if first.vertex_particles is None
                    else list(first.vertex_particles)
                ),
                "coupling": list(first.coupling),
                "helicity_weight": float(first.helicity_weight),
                "row_ids": sorted(root.id for root in roots),
            }
        )
    return {
        "selector_domains_sha256": _canonical_sha256(
            [list(domain) for domain in selector_domains]
        ),
        "current_class_members_sha256": _canonical_sha256(
            [list(members) for members in classes]
        ),
        "current_remap_sha256": _canonical_sha256(list(class_by_current)),
        "interaction_groups_sha256": _canonical_sha256(
            [sorted(row.id for row in rows) for rows in ordered_interaction_groups]
        ),
        "closure_groups_sha256": _canonical_sha256(
            [sorted(root.id for root in roots) for roots in ordered_root_groups]
        ),
        "rectangle_cardinalities_sha256": _canonical_sha256(
            rectangle_cardinalities
        ),
        "row_identity_sha256": _canonical_sha256(row_identities),
    }


def project_rectangular_dynamic_color_classes(
    dag: GenericDAG,
    model: Model,
    *,
    source_revision: str | None = None,
) -> tuple[GenericDAG, DynamicColorProjectionCertificate]:
    """Project exact dynamic-color rectangles into shared sum currents.

    This is the GenericDAG counterpart of topology-replay value projection.
    It is intentionally conservative:

    * source currents remain singleton;
    * every current field except dynamic LC sector remains in the class key;
    * the downstream physical-selector domain is part of that key;
    * concrete kernel identity, coupling, coefficient, and parent order remain
      exact;
    * both transition and amplitude-closure rows must form complete Cartesian
      products without duplicates.

    A failed rectangle splits every implicated class and retries to a fixed
    point.  Therefore a malformed or incomplete color orbit can only reduce
    optimization, never alter the generated amplitude.
    """

    before = _initial_dynamic_color_projection_certificate(
        dag,
        source_revision=source_revision,
        equality_check_status="not-applicable",
    )
    if (
        dag.process.color_accuracy != "lc"
        or not dag.currents
        or not dag.interactions
        or not dag.amplitude_roots
    ):
        return dag, before
    if dag.helicity_recurrence is not None or dag.helicity_materialization is not None:
        raise ValueError(
            "dynamic-color projection must precede helicity recurrence materialization"
        )

    selector_domains = _downstream_color_selector_domains(dag)
    split_current_ids: set[int] = set()
    split_class_count = 0
    while True:
        classes, class_by_current = _projection_partition(
            dag,
            selector_domains,
            split_current_ids,
        )
        interaction_groups, root_groups = _projection_groups(
            dag,
            classes,
            class_by_current,
        )
        invalid_classes = _invalid_projection_classes(
            dag,
            classes,
            class_by_current,
            interaction_groups,
            root_groups,
        )
        newly_split = {
            current_id
            for class_id in invalid_classes
            for current_id in classes[class_id]
            if current_id not in split_current_ids
        }
        if not newly_split:
            break
        split_class_count += len(invalid_classes)
        split_current_ids.update(newly_split)

    proof_payloads = _projection_proof_payloads(
        dag,
        selector_domains,
        classes,
        class_by_current,
        interaction_groups,
        root_groups,
    )
    projected_class_count = sum(len(members) > 1 for members in classes)
    if projected_class_count == 0:
        return dag, replace(
            before,
            selector_domains_sha256=str(
                proof_payloads["selector_domains_sha256"]
            ),
            current_class_members_sha256=str(
                proof_payloads["current_class_members_sha256"]
            ),
            old_to_new_current_ids=class_by_current,
            current_remap_sha256=str(proof_payloads["current_remap_sha256"]),
            interaction_groups_sha256=str(
                proof_payloads["interaction_groups_sha256"]
            ),
            closure_groups_sha256=str(
                proof_payloads["closure_groups_sha256"]
            ),
            rectangle_cardinalities_sha256=str(
                proof_payloads["rectangle_cardinalities_sha256"]
            ),
            row_identity_sha256=str(proof_payloads["row_identity_sha256"]),
            equality_check_status="passed-no-projectable-classes",
            split_current_class_count=split_class_count,
        )

    old_to_new = class_by_current
    currents = tuple(
        replace(dag.currents[members[0]], id=class_id)
        for class_id, members in enumerate(classes)
    )
    sources = tuple(
        old_to_new[source_id]
        for source_id in dag.sources
    )
    if len(sources) != len(set(sources)):
        raise ValueError("dynamic-color projection unexpectedly merged source currents")

    interactions: list[InteractionNode] = []
    rectangular_interaction_count = 0
    for rows in sorted(
        interaction_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    ):
        first = rows[0]
        involved = (
            old_to_new[first.result_id],
            old_to_new[first.left_id],
            old_to_new[first.right_id],
        )
        if any(len(classes[class_id]) > 1 for class_id in involved):
            rectangular_interaction_count += 1
        interactions.append(
            replace(
                first,
                id=len(interactions),
                left_id=old_to_new[first.left_id],
                right_id=old_to_new[first.right_id],
                result_id=old_to_new[first.result_id],
                evaluation_group_id=None,
                evaluation_factor=(1.0, 0.0),
            )
        )

    roots: list[AmplitudeRoot] = []
    rectangular_closure_count = 0
    for grouped_roots in sorted(
        root_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    ):
        first = grouped_roots[0]
        if (
            len(classes[old_to_new[first.left_id]]) > 1
            or len(classes[old_to_new[first.right_id]]) > 1
        ):
            rectangular_closure_count += 1
        roots.append(
            replace(
                first,
                id=len(roots),
                left_id=old_to_new[first.left_id],
                right_id=old_to_new[first.right_id],
                color_sector_id=_resolved_root_color_sector_id(dag, first),
            )
        )

    projected = replace(
        dag,
        currents=currents,
        sources=sources,
        interactions=tuple(interactions),
        amplitude_roots=tuple(roots),
    )
    projected = assign_recursive_current_evaluation_reuse(projected, model)
    certificate = DynamicColorProjectionCertificate(
        abi=_DYNAMIC_COLOR_PROJECTION_ABI,
        source_revision=source_revision,
        source_semantics_sha256=before.source_semantics_sha256,
        selector_domains_sha256=str(
            proof_payloads["selector_domains_sha256"]
        ),
        current_class_members_sha256=str(
            proof_payloads["current_class_members_sha256"]
        ),
        old_to_new_current_ids=old_to_new,
        current_remap_sha256=str(proof_payloads["current_remap_sha256"]),
        interaction_groups_sha256=str(
            proof_payloads["interaction_groups_sha256"]
        ),
        closure_groups_sha256=str(proof_payloads["closure_groups_sha256"]),
        rectangle_cardinalities_sha256=str(
            proof_payloads["rectangle_cardinalities_sha256"]
        ),
        row_identity_sha256=str(proof_payloads["row_identity_sha256"]),
        equality_check_status="passed-exact-cartesian-products-no-duplicates",
        retained_color_metadata=_RETAINED_COLOR_METADATA_POLICY,
        root_sector_policy=_ROOT_SECTOR_POLICY,
        before_current_count=len(dag.currents),
        after_current_count=len(projected.currents),
        before_interaction_count=len(dag.interactions),
        after_interaction_count=len(projected.interactions),
        before_evaluation_count=dag.interaction_evaluation_count,
        after_evaluation_count=projected.interaction_evaluation_count,
        before_amplitude_root_count=len(dag.amplitude_roots),
        after_amplitude_root_count=len(projected.amplitude_roots),
        projected_current_class_count=projected_class_count,
        rectangular_interaction_group_count=rectangular_interaction_count,
        rectangular_closure_group_count=rectangular_closure_count,
        split_current_class_count=split_class_count,
    )
    return projected, certificate


def _derive_current_value_equivalences(
    dag: GenericDAG,
    model: Model,
    *,
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] | None = None,
    discovery: _NumericalRelationDiscovery | None = None,
) -> tuple[_CurrentValueEquivalence, ...]:
    """Derive current classes in increasing external-subset order."""

    kernel_equivalences = {} if equivalence_by_kind is None else equivalence_by_kind
    interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
    for interaction in dag.interactions:
        interactions_by_result[interaction.result_id].append(interaction)

    current_equivalences: list[_CurrentValueEquivalence | None] = [None] * len(
        dag.currents
    )
    source_representative_by_key: dict[tuple[object, ...], int] = {}
    equivalence_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
    ] = {}
    projective_representatives_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector],
        list[_ProjectiveExpressionRepresentative],
    ] = {}

    ordered_currents = sorted(
        dag.currents,
        key=lambda current: (len(current.index.external_labels), current.id),
    )
    for current in ordered_currents:
        contract = _current_evaluation_contract(current)
        if current.is_source:
            if current.source_leg_label is None or current.source_helicity is None:
                raise ValueError(
                    f"source current {current.id} lacks physical source metadata"
                )
            source_key = (
                contract,
                int(current.source_leg_label),
                int(current.source_helicity),
            )
            representative_id = source_representative_by_key.setdefault(
                source_key,
                current.id,
            )
            current_equivalences[current.id] = _CurrentValueEquivalence(
                representative_id=representative_id,
                factor=(1.0, 0.0),
            )
            continue

        term_vector = _current_term_vector(
            dag,
            current,
            interactions_by_result[current.id],
            model,
            current_equivalences=current_equivalences,
            equivalence_by_kind=kernel_equivalences,
        )
        equivalence = _classify_current_term_vector(
            current_id=current.id,
            contract=contract,
            term_vector=term_vector,
            equivalence_by_expression=equivalence_by_expression,
            projective_representatives_by_expression=(
                projective_representatives_by_expression
            ),
        )
        if discovery is not None:
            equivalence = discovery.consider(
                current_id=current.id,
                contract=contract,
                term_vector=term_vector,
                baseline=equivalence,
            )
        current_equivalences[current.id] = equivalence

    if any(item is None for item in current_equivalences):
        raise ValueError(
            "current-value equivalence derivation left an unclassified current"
        )
    return tuple(item for item in current_equivalences if item is not None)


def verify_dag_relation_certificates(
    dag: GenericDAG,
    model: Model,
    certificates: Iterable[ExactCurrentRelationCertificate],
) -> bool:
    """Replay a certificate chain without consulting numerical probe results."""

    ordered_certificates = tuple(certificates)
    if (
        tuple(sorted(ordered_certificates, key=lambda item: item.current_id))
        != ordered_certificates
    ):
        return False
    certificate_by_current = {
        certificate.current_id: certificate for certificate in ordered_certificates
    }
    if len(certificate_by_current) != len(ordered_certificates):
        return False

    try:
        interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
        for interaction in dag.interactions:
            interactions_by_result[interaction.result_id].append(interaction)
        current_equivalences: list[_CurrentValueEquivalence | None] = [None] * len(
            dag.currents
        )
        source_representative_by_key: dict[tuple[object, ...], int] = {}
        equivalence_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
        ] = {}
        projective_representatives_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector],
            list[_ProjectiveExpressionRepresentative],
        ] = {}
        kernel_equivalences: dict[int, VertexEvaluationEquivalence] = {}
        term_vector_by_current: dict[int, _CurrentTermVector] = {}
        contract_by_current: dict[int, _CurrentContract] = {}
        consumed: set[int] = set()

        ordered_currents = sorted(
            dag.currents,
            key=lambda current: (len(current.index.external_labels), current.id),
        )
        for current in ordered_currents:
            contract = _current_evaluation_contract(current)
            contract_by_current[current.id] = contract
            if current.is_source:
                if current.id in certificate_by_current:
                    return False
                if current.source_leg_label is None or current.source_helicity is None:
                    return False
                source_key = (
                    contract,
                    int(current.source_leg_label),
                    int(current.source_helicity),
                )
                representative_id = source_representative_by_key.setdefault(
                    source_key,
                    current.id,
                )
                current_equivalences[current.id] = _CurrentValueEquivalence(
                    representative_id,
                    (1.0, 0.0),
                )
                continue

            term_vector = _current_term_vector(
                dag,
                current,
                interactions_by_result[current.id],
                model,
                current_equivalences=current_equivalences,
                equivalence_by_kind=kernel_equivalences,
            )
            term_vector_by_current[current.id] = term_vector
            equivalence = _classify_current_term_vector(
                current_id=current.id,
                contract=contract,
                term_vector=term_vector,
                equivalence_by_expression=equivalence_by_expression,
                projective_representatives_by_expression=(
                    projective_representatives_by_expression
                ),
            )
            certificate = certificate_by_current.get(current.id)
            if certificate is not None:
                if (
                    certificate.algorithm != RELATION_DISCOVERY_CERTIFICATE_ALGORITHM
                    or certificate.representative_id >= current.id
                    or certificate.representative_id not in term_vector_by_current
                    or contract_by_current[certificate.representative_id] != contract
                ):
                    return False
                representative_vector = term_vector_by_current[
                    certificate.representative_id
                ]
                factor = _certify_exact_term_vector_relation(
                    representative_vector,
                    term_vector,
                )
                if factor is None or not _complex_weight_bits_equal(
                    factor,
                    certificate.factor,
                ):
                    return False
                if (
                    _term_vector_sha256(term_vector)
                    != certificate.current_term_vector_sha256
                    or _term_vector_sha256(representative_vector)
                    != certificate.representative_term_vector_sha256
                    or _certificate_proof_sha256(
                        current_id=certificate.current_id,
                        representative_id=certificate.representative_id,
                        factor=certificate.factor,
                        current_term_vector_sha256=(
                            certificate.current_term_vector_sha256
                        ),
                        representative_term_vector_sha256=(
                            certificate.representative_term_vector_sha256
                        ),
                    )
                    != certificate.proof_sha256
                ):
                    return False
                equivalence = _CurrentValueEquivalence(
                    certificate.representative_id,
                    certificate.factor,
                )
                consumed.add(current.id)
            current_equivalences[current.id] = equivalence
        return consumed == set(certificate_by_current)
    except (IndexError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _numerical_projective_fingerprint(
    term_vector: _CurrentTermVector,
    *,
    precision_digits: int,
    probe_count: int,
    seed: int,
) -> tuple[object, ...] | None:
    """Return a scale-free high-precision candidate key.

    This fingerprint is deliberately insufficient as a proof. Its only use is
    to reduce the number of pairs sent to exact coefficient-vector replay.
    """

    if not _term_vector_is_finite_nonzero(term_vector):
        return None
    with localcontext() as context:
        context.prec = precision_digits
        values: list[tuple[Decimal, Decimal]] = []
        for probe_index in range(probe_count):
            total_real = Decimal(0)
            total_imaginary = Decimal(0)
            for evaluation_key, coefficient in term_vector:
                probe_real, probe_imaginary = _deterministic_decimal_probe_value(
                    evaluation_key,
                    seed=seed,
                    probe_index=probe_index,
                )
                coefficient_real = Decimal.from_float(coefficient[0])
                coefficient_imaginary = Decimal.from_float(coefficient[1])
                total_real += (
                    coefficient_real * probe_real
                    - coefficient_imaginary * probe_imaginary
                )
                total_imaginary += (
                    coefficient_real * probe_imaginary
                    + coefficient_imaginary * probe_real
                )
            values.append((total_real, total_imaginary))

        pivot_index = next(
            (
                index
                for index, value in enumerate(values)
                if value != (Decimal(0), Decimal(0))
            ),
            None,
        )
        if pivot_index is None:
            return None
        pivot = values[pivot_index]
        normalized = tuple(_decimal_complex_ratio(value, pivot) for value in values)
        if any(value is None for value in normalized):
            return None
        fingerprint_digits = max(24, precision_digits // 2)
        return (
            "deterministic-projective-probe-v1",
            pivot_index,
            tuple(
                (
                    _decimal_fingerprint_component(value[0], fingerprint_digits),
                    _decimal_fingerprint_component(value[1], fingerprint_digits),
                )
                for value in normalized
                if value is not None
            ),
        )


def _deterministic_decimal_probe_value(
    evaluation_key: _EvaluationKey,
    *,
    seed: int,
    probe_index: int,
) -> tuple[Decimal, Decimal]:
    payload = json.dumps(
        _canonical_proof_value(evaluation_key),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(f"{seed}:{probe_index}:".encode("ascii") + payload).digest()

    def component(chunk: bytes) -> Decimal:
        raw = int.from_bytes(chunk, "big", signed=False)
        value = raw % 2_000_003 - 1_000_001
        return Decimal(1 if value == 0 else value)

    return component(digest[:8]), component(digest[8:16])


def _decimal_complex_ratio(
    numerator: tuple[Decimal, Decimal],
    denominator: tuple[Decimal, Decimal],
) -> tuple[Decimal, Decimal] | None:
    denominator_norm = denominator[0] * denominator[0] + denominator[1] * denominator[1]
    if denominator_norm == 0:
        return None
    return (
        (numerator[0] * denominator[0] + numerator[1] * denominator[1])
        / denominator_norm,
        (numerator[1] * denominator[0] - numerator[0] * denominator[1])
        / denominator_norm,
    )


def _decimal_fingerprint_component(value: Decimal, digits: int) -> str:
    if value == 0:
        return "0"
    return format(value, f".{digits}E")


def _certify_exact_term_vector_relation(
    representative: _CurrentTermVector,
    candidate: _CurrentTermVector,
) -> _ComplexWeight | None:
    if (
        not _term_vector_is_finite_nonzero(representative)
        or not _term_vector_is_finite_nonzero(candidate)
        or len(representative) != len(candidate)
    ):
        return None
    if any(
        representative_key != candidate_key
        for (representative_key, _), (candidate_key, _) in zip(
            representative,
            candidate,
            strict=True,
        )
    ):
        return None
    factor = _exact_representable_complex_ratio(
        candidate[0][1],
        representative[0][1],
    )
    if factor is None or not _term_vector_scaled_exactly(
        representative,
        factor,
        candidate,
    ):
        return None
    return factor


def _build_exact_current_relation_certificate(
    *,
    current_id: int,
    representative_id: int,
    factor: _ComplexWeight,
    current_term_vector: _CurrentTermVector,
    representative_term_vector: _CurrentTermVector,
) -> ExactCurrentRelationCertificate:
    current_digest = _term_vector_sha256(current_term_vector)
    representative_digest = _term_vector_sha256(representative_term_vector)
    return ExactCurrentRelationCertificate(
        current_id=current_id,
        representative_id=representative_id,
        factor=factor,
        current_term_vector_sha256=current_digest,
        representative_term_vector_sha256=representative_digest,
        proof_sha256=_certificate_proof_sha256(
            current_id=current_id,
            representative_id=representative_id,
            factor=factor,
            current_term_vector_sha256=current_digest,
            representative_term_vector_sha256=representative_digest,
        ),
    )


def _certificate_proof_sha256(
    *,
    current_id: int,
    representative_id: int,
    factor: _ComplexWeight,
    current_term_vector_sha256: str,
    representative_term_vector_sha256: str,
) -> str:
    return _canonical_payload_sha256(
        {
            "algorithm": RELATION_DISCOVERY_CERTIFICATE_ALGORITHM,
            "current_id": current_id,
            "representative_id": representative_id,
            "factor_binary64": [factor[0].hex(), factor[1].hex()],
            "current_term_vector_sha256": current_term_vector_sha256,
            "representative_term_vector_sha256": (representative_term_vector_sha256),
        }
    )


def _term_vector_sha256(term_vector: _CurrentTermVector) -> str:
    return _canonical_payload_sha256(
        [
            {
                "evaluation_key": _canonical_proof_value(evaluation_key),
                "coefficient_binary64": [
                    coefficient[0].hex(),
                    coefficient[1].hex(),
                ],
            }
            for evaluation_key, coefficient in term_vector
        ]
    )


def _canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_proof_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_canonical_proof_value(item) for item in value]
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(
        "relation certificate contains a noncanonical evaluation-key value "
        f"{type(value).__name__}"
    )


def _interaction_reuse_signature(dag: GenericDAG) -> tuple[tuple[object, ...], ...]:
    canonical_group_by_key: dict[tuple[str, int], int] = {}
    signature: list[tuple[object, ...]] = []
    for interaction in dag.interactions:
        raw_group = (
            ("group", interaction.evaluation_group_id)
            if interaction.evaluation_group_id is not None
            else ("interaction", interaction.id)
        )
        canonical_group = canonical_group_by_key.setdefault(
            raw_group,
            len(canonical_group_by_key),
        )
        signature.append(
            (
                canonical_group,
                interaction.evaluation_factor[0].hex(),
                interaction.evaluation_factor[1].hex(),
            )
        )
    return tuple(signature)


def _classify_current_term_vector(
    *,
    current_id: int,
    contract: _CurrentContract,
    term_vector: _CurrentTermVector,
    equivalence_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
    ],
    projective_representatives_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector],
        list[_ProjectiveExpressionRepresentative],
    ],
) -> _CurrentValueEquivalence:
    """Return the earliest exactly proven representative for one current.

    Exact equality and sign reuse remain the fast paths.  More general
    projective reuse is accepted only when a finite, nonzero binary64 factor
    exists exactly and applying it reproduces every coefficient bit-for-bit.
    """

    identity = _CurrentValueEquivalence(current_id, (1.0, 0.0))
    expression_key = (contract, term_vector)
    if not term_vector:
        exact_zero = equivalence_by_expression.get(expression_key)
        if exact_zero is not None:
            return exact_zero
        equivalence_by_expression[expression_key] = identity
        return identity
    if not _term_vector_is_finite_nonzero(term_vector):
        return identity

    exact = equivalence_by_expression.get(expression_key)
    if exact is not None:
        return exact

    opposite = equivalence_by_expression.get(
        (contract, _negate_term_vector(term_vector))
    )
    if opposite is not None:
        opposite_factor = (
            _canonical_zero(-opposite.factor[0]),
            _canonical_zero(-opposite.factor[1]),
        )
        representative_vector = _term_vector_divided_by_exact_factor(
            term_vector,
            opposite_factor,
        )
        if representative_vector is not None and _term_vector_scaled_exactly(
            representative_vector,
            opposite_factor,
            term_vector,
        ):
            equivalence = _CurrentValueEquivalence(
                opposite.representative_id,
                opposite_factor,
            )
            equivalence_by_expression[expression_key] = equivalence
            return equivalence

    canonical = _canonicalize_projective_term_vector(term_vector)
    if canonical is None:
        equivalence_by_expression[expression_key] = identity
        return identity

    projective_key = (contract, canonical.term_vector)
    representatives = projective_representatives_by_expression.get(projective_key)
    matches: list[_CurrentValueEquivalence] = []
    if representatives is not None:
        for representative in representatives:
            factor = _exact_representable_complex_ratio(
                canonical.factor,
                representative.normalization_factor,
            )
            if factor is None or not _term_vector_scaled_exactly(
                representative.term_vector,
                factor,
                term_vector,
            ):
                continue
            matches.append(
                _CurrentValueEquivalence(
                    representative.representative_id,
                    factor,
                )
            )

    if len(matches) == 1:
        equivalence = matches[0]
        equivalence_by_expression[expression_key] = equivalence
        return equivalence

    # Zero matches mean that the normalized lookup key was only a rounded
    # collision.  Multiple matches make the concrete representative
    # ambiguous.  Both cases fail closed by retaining this current.
    projective_representatives_by_expression.setdefault(
        projective_key,
        [],
    ).append(
        _ProjectiveExpressionRepresentative(
            representative_id=current_id,
            term_vector=term_vector,
            normalization_factor=canonical.factor,
        )
    )
    equivalence_by_expression[expression_key] = identity
    return identity


def _canonicalize_projective_term_vector(
    term_vector: _CurrentTermVector,
) -> _CanonicalProjectiveTermVector | None:
    """Normalize a finite nonzero vector for projective-class lookup.

    The first canonical term is the deterministic pivot.  The normalized
    binary64 coefficients are only an index: every accepted equivalence is
    separately proven with an exactly representable factor and bit-exact
    coefficient reconstruction.
    """

    if not term_vector or not _term_vector_is_finite_nonzero(term_vector):
        return None
    keys = tuple(key for key, _coefficient in term_vector)
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        return None

    factor = _canonical_complex_weight(term_vector[0][1])
    normalized_terms: list[tuple[_EvaluationKey, _ComplexWeight]] = []
    for key, raw_coefficient in term_vector:
        coefficient = _canonical_complex_weight(raw_coefficient)
        normalized = _roundtrip_complex_ratio(coefficient, factor)
        if normalized is None:
            return None
        normalized_terms.append((key, normalized))
    return _CanonicalProjectiveTermVector(
        term_vector=tuple(normalized_terms),
        factor=factor,
    )


def _roundtrip_complex_ratio(
    numerator: _ComplexWeight,
    denominator: _ComplexWeight,
) -> _ComplexWeight | None:
    """Return a finite lookup ratio that reconstructs ``numerator`` exactly."""

    if (
        not _complex_weight_is_finite(numerator)
        or not _complex_weight_is_finite(denominator)
        or denominator == (0.0, 0.0)
    ):
        return None
    try:
        quotient = complex(*numerator) / complex(*denominator)
    except (OverflowError, ZeroDivisionError):
        return None
    result = _canonical_complex_weight((quotient.real, quotient.imag))
    if not _complex_weight_is_finite(result):
        return None
    reconstructed = _canonical_complex_weight(_complex_weight_mul(denominator, result))
    if not _complex_weight_bits_equal(reconstructed, numerator):
        return None
    return result


def _exact_representable_complex_ratio(
    numerator: _ComplexWeight,
    denominator: _ComplexWeight,
) -> _ComplexWeight | None:
    """Return the exact binary64 complex quotient, or fail closed.

    ``Fraction`` is used only after two vectors collide on their inexpensive
    normalized key.  It proves that both quotient components are exactly
    representable rather than merely rounded values that happen to be close.
    """

    if (
        not _complex_weight_is_finite(numerator)
        or not _complex_weight_is_finite(denominator)
        or denominator == (0.0, 0.0)
    ):
        return None
    numerator_real = Fraction.from_float(numerator[0])
    numerator_imag = Fraction.from_float(numerator[1])
    denominator_real = Fraction.from_float(denominator[0])
    denominator_imag = Fraction.from_float(denominator[1])
    denominator_norm = (
        denominator_real * denominator_real + denominator_imag * denominator_imag
    )
    if denominator_norm == 0:
        return None
    real = (
        numerator_real * denominator_real + numerator_imag * denominator_imag
    ) / denominator_norm
    imaginary = (
        numerator_imag * denominator_real - numerator_real * denominator_imag
    ) / denominator_norm
    real_f64 = _exact_fraction_as_f64(real)
    imaginary_f64 = _exact_fraction_as_f64(imaginary)
    if real_f64 is None or imaginary_f64 is None:
        return None
    factor = (real_f64, imaginary_f64)
    reconstructed = _canonical_complex_weight(_complex_weight_mul(denominator, factor))
    if not _complex_weight_bits_equal(
        reconstructed,
        _canonical_complex_weight(numerator),
    ):
        return None
    return factor


def _exact_representable_complex_product(
    left: _ComplexWeight,
    right: _ComplexWeight,
) -> _ComplexWeight | None:
    """Return the exact binary64 complex product, or fail closed."""

    if not _complex_weight_is_finite(left) or not _complex_weight_is_finite(right):
        return None
    left_real = Fraction.from_float(left[0])
    left_imaginary = Fraction.from_float(left[1])
    right_real = Fraction.from_float(right[0])
    right_imaginary = Fraction.from_float(right[1])
    real = left_real * right_real - left_imaginary * right_imaginary
    imaginary = left_real * right_imaginary + left_imaginary * right_real
    real_f64 = _exact_fraction_as_f64(real)
    imaginary_f64 = _exact_fraction_as_f64(imaginary)
    if real_f64 is None or imaginary_f64 is None:
        return None
    product = (real_f64, imaginary_f64)
    reconstructed = _canonical_complex_weight(_complex_weight_mul(left, right))
    if not _complex_weight_bits_equal(reconstructed, product):
        return None
    return product


def _exact_fraction_as_f64(value: Fraction) -> float | None:
    try:
        result = float(value)
    except OverflowError:
        return None
    if not isfinite(result) or Fraction.from_float(result) != value:
        return None
    return _canonical_zero(result)


def _term_vector_scaled_exactly(
    representative: _CurrentTermVector,
    factor: _ComplexWeight,
    candidate: _CurrentTermVector,
) -> bool:
    if (
        not _complex_weight_is_finite(factor)
        or factor == (0.0, 0.0)
        or len(representative) != len(candidate)
    ):
        return False
    for (representative_key, representative_value), (
        candidate_key,
        candidate_value,
    ) in zip(representative, candidate, strict=True):
        if representative_key != candidate_key:
            return False
        scaled = _canonical_complex_weight(
            _complex_weight_mul(factor, representative_value)
        )
        if not _complex_weight_bits_equal(
            scaled,
            _canonical_complex_weight(candidate_value),
        ):
            return False
    return True


def _term_vector_divided_by_exact_factor(
    candidate: _CurrentTermVector,
    factor: _ComplexWeight,
) -> _CurrentTermVector | None:
    """Recover a representative vector only for the sign fast path."""

    if factor not in ((1.0, 0.0), (-1.0, 0.0)):
        return None
    return tuple(
        (
            key,
            value
            if factor == (1.0, 0.0)
            else (_canonical_zero(-value[0]), _canonical_zero(-value[1])),
        )
        for key, value in candidate
    )


def _term_vector_is_finite_nonzero(term_vector: _CurrentTermVector) -> bool:
    return bool(term_vector) and all(
        coefficient != (0.0, 0.0) and _complex_weight_is_finite(coefficient)
        for _key, coefficient in term_vector
    )


def _complex_weight_is_finite(value: _ComplexWeight) -> bool:
    return isfinite(value[0]) and isfinite(value[1])


def _canonical_complex_weight(value: _ComplexWeight) -> _ComplexWeight:
    return (_canonical_zero(value[0]), _canonical_zero(value[1]))


def _complex_weight_bits_equal(
    left: _ComplexWeight,
    right: _ComplexWeight,
) -> bool:
    return left[0].hex() == right[0].hex() and left[1].hex() == right[1].hex()


def _current_evaluation_contract(current: CurrentNode) -> _CurrentContract:
    """Return fields that can affect source, kernel, or propagator values."""

    index = current.index
    return (
        int(index.particle_id),
        int(index.external_mask),
        index.external_labels,
        int(index.chirality),
        index.spin_state,
        index.flavour_flow,
        index.quantum_number_flow,
        int(index.momentum_mask),
        index.coupling_orders,
        index.auxiliary_kind,
        int(current.dimension),
        bool(current.is_source),
    )


def _current_term_vector(
    dag: GenericDAG,
    current: CurrentNode,
    interactions: list[InteractionNode],
    model: Model,
    *,
    current_equivalences: list[_CurrentValueEquivalence | None],
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence],
) -> _CurrentTermVector:
    coefficients_by_key: dict[_EvaluationKey, list[_ComplexWeight]] = defaultdict(list)
    for interaction in interactions:
        kernel_equivalence = _kernel_equivalence(
            model,
            interaction.vertex_kind,
            equivalence_by_kind,
        )
        left = current_equivalences[interaction.left_id]
        right = current_equivalences[interaction.right_id]
        if left is None or right is None:
            raise ValueError(
                "current-value equivalence requires parents from an earlier subset"
            )
        canonical_inputs, evaluation_factor = _canonical_interaction_evaluation(
            kernel_equivalence,
            left_id=interaction.left_id,
            right_id=interaction.right_id,
            left=left,
            right=right,
        )
        term_key = (
            kernel_equivalence.class_id,
            canonical_inputs,
            int(current.index.particle_id),
            int(current.index.chirality),
            _runtime_coupling_identity(
                model,
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                coupling=interaction.coupling,
            ),
        )
        coefficient = _complex_weight_mul(
            interaction.color_weight,
            evaluation_factor,
        )
        coefficients_by_key[term_key].append(coefficient)

    terms: list[tuple[_EvaluationKey, _ComplexWeight]] = []
    for grouped_term_key in sorted(coefficients_by_key):
        coefficients = coefficients_by_key[grouped_term_key]
        coefficient = (
            _canonical_zero(fsum(value[0] for value in coefficients)),
            _canonical_zero(fsum(value[1] for value in coefficients)),
        )
        if coefficient != (0.0, 0.0):
            terms.append((grouped_term_key, coefficient))
    return tuple(terms)


def _kernel_equivalence(
    model: Model,
    kind: int,
    cache: dict[int, VertexEvaluationEquivalence],
) -> VertexEvaluationEquivalence:
    cached = cache.get(kind)
    if cached is not None:
        return cached
    equivalence = model.vertex_evaluation_equivalence(kind)
    if not equivalence.verified:
        model_type = f"{type(model).__module__}.{type(model).__qualname__}"
        equivalence = VertexEvaluationEquivalence(class_id=f"{model_type}:{int(kind)}")
    cache[kind] = equivalence
    return equivalence


def _runtime_coupling_identity(
    model: Model,
    *,
    vertex_kind: int,
    vertex_particles: tuple[int, int, int],
    coupling: tuple[float, float],
) -> tuple[tuple[float, float], tuple[tuple[int, str], ...]]:
    """Return defaults plus stable mutable-parameter provenance for reuse."""

    names = runtime_coupling_parameter_names(
        vertex_kind,
        vertex_particles,
        coupling,
        model=model,
    )
    provenance = tuple((0, "") if name is None else (1, str(name)) for name in names)
    return coupling, provenance


def _negate_term_vector(vector: _CurrentTermVector) -> _CurrentTermVector:
    return tuple((key, (-value[0], -value[1])) for key, value in vector)


def _complex_weight_mul(
    left: _ComplexWeight,
    right: _ComplexWeight,
) -> _ComplexWeight:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


__all__ = [
    "RELATION_DISCOVERY_CERTIFICATE_ALGORITHM",
    "RELATION_DISCOVERY_SCHEMA_VERSION",
    "DynamicColorProjectionCertificate",
    "ExactCurrentRelationCertificate",
    "RecursiveEvaluationReuseTracker",
    "RelationDiscoveryReport",
    "RelationDiscoveryResult",
    "assign_recursive_current_evaluation_reuse",
    "discover_recursive_evaluation_relations",
    "project_rectangular_dynamic_color_classes",
    "recurrence_relation_discovery_diagnostic",
    "verify_dag_relation_certificates",
]
