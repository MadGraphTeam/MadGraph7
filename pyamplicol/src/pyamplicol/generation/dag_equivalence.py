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
from fractions import Fraction
from math import fsum, isfinite
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
        current_equivalences[current.id] = _classify_current_term_vector(
            current_id=current.id,
            contract=contract,
            term_vector=term_vector,
            equivalence_by_expression=equivalence_by_expression,
            projective_representatives_by_expression=(
                projective_representatives_by_expression
            ),
        )

    if any(item is None for item in current_equivalences):
        raise ValueError(
            "current-value equivalence derivation left an unclassified current"
        )
    return tuple(item for item in current_equivalences if item is not None)


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
    "RecursiveEvaluationReuseTracker",
    "assign_recursive_current_evaluation_reuse",
]
