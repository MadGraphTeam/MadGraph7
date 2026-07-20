# SPDX-License-Identifier: 0BSD
"""Proof-gated recursive current-value reuse for generated DAGs.

Current identity includes colour-sector and ordering metadata needed to build
and reduce amplitudes.  Those fields do not necessarily change the numerical
current value.  This module proves such value equivalences from the complete
recursive computation instead of guessing them from particle names or PDGs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import fsum
from typing import TypeAlias

from ..models.base import Model, VertexEvaluationEquivalence
from .contracts import runtime_coupling_parameter_names
from .dag_types import CurrentNode, GenericDAG, InteractionNode

_ComplexWeight: TypeAlias = tuple[float, float]
_CurrentContract: TypeAlias = tuple[object, ...]
_EvaluationKey: TypeAlias = tuple[object, ...]
_CurrentTermVector: TypeAlias = tuple[tuple[_EvaluationKey, _ComplexWeight], ...]


@dataclass(frozen=True, slots=True)
class _CurrentValueEquivalence:
    """Exact relation ``current = factor * representative``."""

    representative_id: int
    factor: _ComplexWeight


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
        self._representative_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector], int
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
        canonical_inputs, kernel_factor = _canonical_kernel_evaluation(
            kernel_equivalence,
            left.representative_id,
            right.representative_id,
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
        evaluation_factor = _complex_weight_mul(
            kernel_factor,
            _complex_weight_mul(left.factor, right.factor),
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
                    coefficient = coefficients
                if coefficient != (0.0, 0.0):
                    terms.append(((group_id,), coefficient))
            term_vector = tuple(terms)
            contract = _current_evaluation_contract(current)
            expression_key = (contract, term_vector)
            representative_id = self._representative_by_expression.get(expression_key)
            factor: _ComplexWeight = (1.0, 0.0)
            if representative_id is None:
                opposite_key = (contract, _negate_term_vector(term_vector))
                representative_id = self._representative_by_expression.get(opposite_key)
                if representative_id is not None:
                    factor = (-1.0, 0.0)
            if representative_id is None:
                representative_id = current.id
                self._representative_by_expression[expression_key] = representative_id
            self._append_current_equivalence(
                current,
                _CurrentValueEquivalence(representative_id, factor),
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


def assign_recursive_current_evaluation_reuse(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Share kernel evaluations through exactly proven current equivalences.

    The proof is recursive.  Duplicate source wavefunctions form the base
    classes.  A generated current joins an existing class only when its full
    vector of model-certified kernel terms and coefficients is byte-exactly
    equal or opposite to the representative vector.  The current contract
    keeps every field consumed by source, kernel, and propagator evaluation;
    only colour bookkeeping, ordering metadata, and ancestry bit allocation
    are deliberately excluded.

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
        canonical_inputs, kernel_factor = _canonical_kernel_evaluation(
            kernel_equivalence,
            left.representative_id,
            right.representative_id,
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
        input_factor = _complex_weight_mul(left.factor, right.factor)
        evaluation_factor = _complex_weight_mul(
            kernel_factor,
            input_factor,
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


def _derive_current_value_equivalences(
    dag: GenericDAG,
    model: Model,
    *,
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] | None = None,
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
    representative_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector], int
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
        expression_key = (contract, term_vector)
        current_representative_id = representative_by_expression.get(expression_key)
        factor: _ComplexWeight = (1.0, 0.0)
        if current_representative_id is None:
            opposite_key = (contract, _negate_term_vector(term_vector))
            current_representative_id = representative_by_expression.get(opposite_key)
            if current_representative_id is not None:
                factor = (-1.0, 0.0)
        if current_representative_id is None:
            current_representative_id = current.id
            representative_by_expression[expression_key] = current_representative_id
        current_equivalences[current.id] = _CurrentValueEquivalence(
            representative_id=current_representative_id,
            factor=factor,
        )

    if any(item is None for item in current_equivalences):
        raise ValueError(
            "current-value equivalence derivation left an unclassified current"
        )
    return tuple(item for item in current_equivalences if item is not None)


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
        canonical_inputs, kernel_factor = _canonical_kernel_evaluation(
            kernel_equivalence,
            left.representative_id,
            right.representative_id,
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
        input_factor = _complex_weight_mul(left.factor, right.factor)
        coefficient = _complex_weight_mul(
            interaction.color_weight,
            _complex_weight_mul(kernel_factor, input_factor),
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
    "RecursiveEvaluationReuseTracker",
    "assign_recursive_current_evaluation_reuse",
]
