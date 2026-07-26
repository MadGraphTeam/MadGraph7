# SPDX-License-Identifier: 0BSD
"""Algebraically exact projective-current proof contracts."""

from __future__ import annotations

from dataclasses import replace
from math import inf

from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    _canonicalize_projective_term_vector,
    _classify_current_term_vector,
    _complex_weight_mul,
    _exact_representable_complex_product,
    _exact_representable_complex_ratio,
    _term_vector_scaled_exactly,
    assign_recursive_current_evaluation_reuse,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _vector(
    *coefficients: tuple[float, float],
) -> tuple[tuple[tuple[object, ...], tuple[float, float]], ...]:
    return tuple(
        ((index,), coefficient) for index, coefficient in enumerate(coefficients)
    )


def _classify_sequence(
    vectors: tuple[
        tuple[tuple[tuple[object, ...], tuple[float, float]], ...],
        ...,
    ],
):
    equivalence_by_expression = {}
    projective_representatives_by_expression = {}
    return tuple(
        _classify_current_term_vector(
            current_id=current_id,
            contract=("same-current-contract",),
            term_vector=term_vector,
            equivalence_by_expression=equivalence_by_expression,
            projective_representatives_by_expression=(
                projective_representatives_by_expression
            ),
        )
        for current_id, term_vector in enumerate(vectors)
    )


def test_projective_current_reuse_retains_an_exact_complex_factor() -> None:
    representative = _vector((2.0, 0.0), (0.0, 4.0))
    scaled = _vector((0.0, 1.0), (-2.0, 0.0))

    equivalences = _classify_sequence((representative, scaled, scaled))

    assert equivalences[0].representative_id == 0
    assert equivalences[0].factor == (1.0, 0.0)
    assert equivalences[1].representative_id == 0
    assert equivalences[1].factor == (0.0, 0.5)
    assert equivalences[2] == equivalences[1]
    assert _term_vector_scaled_exactly(
        representative,
        equivalences[1].factor,
        scaled,
    )


def test_identity_and_sign_reuse_remain_deterministic_fast_paths() -> None:
    representative = _vector((2.0, 1.0), (-4.0, 3.0))
    opposite = _vector((-2.0, -1.0), (4.0, -3.0))
    vectors = (representative, representative, opposite)

    first = _classify_sequence(vectors)
    second = _classify_sequence(vectors)

    assert first == second
    assert tuple(item.representative_id for item in first) == (0, 0, 0)
    assert tuple(item.factor for item in first) == (
        (1.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
    )


def test_exact_zero_vectors_reuse_without_projective_normalization() -> None:
    equivalences = _classify_sequence(((), ()))

    assert tuple(item.representative_id for item in equivalences) == (0, 0)
    assert tuple(item.factor for item in equivalences) == (
        (1.0, 0.0),
        (1.0, 0.0),
    )
    assert _canonicalize_projective_term_vector(()) is None


def test_projective_reuse_rejects_a_nonproportional_coefficient() -> None:
    representative = _vector((2.0, 0.0), (0.0, 4.0))
    deformed = _vector((0.0, 1.0), (-2.5, 0.0))

    equivalences = _classify_sequence((representative, deformed))

    assert equivalences[1].representative_id == 1
    assert equivalences[1].factor == (1.0, 0.0)


def test_projective_reuse_rejects_a_rounded_unrepresentable_factor() -> None:
    representative = _vector((3.0, 0.0), (6.0, 0.0))
    rounded_third = _vector((1.0, 0.0), (2.0, 0.0))

    assert (
        _exact_representable_complex_ratio(
            rounded_third[0][1],
            representative[0][1],
        )
        is None
    )
    equivalences = _classify_sequence((representative, rounded_third))
    assert equivalences[1].representative_id == 1


def test_recursive_projective_factor_composition_fails_closed() -> None:
    assert _exact_representable_complex_product(
        (2.0, 0.0),
        (0.5, 0.0),
    ) == (1.0, 0.0)
    assert (
        _exact_representable_complex_product(
            (1.0e308, 0.0),
            (1.0e308, 0.0),
        )
        is None
    )
    assert (
        _exact_representable_complex_product(
            (0.1, 0.0),
            (0.2, 0.0),
        )
        is None
    )


def test_projective_reuse_rejects_an_ambiguous_exact_representative() -> None:
    first = _vector((3.0, 0.0), (6.0, 0.0))
    second = _vector((1.0, 0.0), (2.0, 0.0))
    exact_multiple_of_both = _vector((6.0, 0.0), (12.0, 0.0))

    equivalences = _classify_sequence((first, second, exact_multiple_of_both))

    assert equivalences[0].representative_id == 0
    assert equivalences[1].representative_id == 1
    assert equivalences[2].representative_id == 2


def test_projective_canonicalization_rejects_unsafe_vectors() -> None:
    assert _canonicalize_projective_term_vector(()) is None
    assert _canonicalize_projective_term_vector(_vector((0.0, 0.0))) is None
    assert _canonicalize_projective_term_vector(_vector((inf, 0.0))) is None

    duplicate_key = (
        ((0,), (1.0, 0.0)),
        ((0,), (2.0, 0.0)),
    )
    unordered = (
        ((1,), (1.0, 0.0)),
        ((0,), (2.0, 0.0)),
    )
    assert _canonicalize_projective_term_vector(duplicate_key) is None
    assert _canonicalize_projective_term_vector(unordered) is None


def test_projective_factor_propagates_to_a_child_evaluation_group() -> None:
    """A proportional parent must share its child's kernel, not the root value."""

    model = BuiltinSMModel()
    seed = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="lc"),
        model=model,
    )
    template = seed.interactions[0]
    generated_template = seed.currents[template.result_id]
    source_templates = tuple(seed.currents[source_id] for source_id in seed.sources[:2])
    sources = tuple(
        replace(source, id=current_id)
        for current_id, source in enumerate(source_templates)
    )
    generated = tuple(
        replace(generated_template, id=current_id) for current_id in range(2, 6)
    )

    def interaction(
        interaction_id: int,
        left_id: int,
        right_id: int,
        result_id: int,
        color_weight: tuple[float, float],
    ):
        return replace(
            template,
            id=interaction_id,
            left_id=left_id,
            right_id=right_id,
            result_id=result_id,
            color_weight=color_weight,
            evaluation_group_id=None,
            evaluation_factor=(1.0, 0.0),
        )

    interactions = (
        interaction(0, 0, 1, 2, (1.0, 0.0)),
        interaction(1, 0, 1, 3, (2.0, 0.0)),
        interaction(2, 2, 1, 4, (1.0, 0.0)),
        interaction(3, 3, 1, 5, (1.0, 0.0)),
    )
    root = replace(
        seed.amplitude_roots[0],
        id=0,
        left_id=5,
        right_id=1,
    )
    dag = replace(
        seed,
        currents=(*sources, *generated),
        sources=(0, 1),
        interactions=interactions,
        amplitude_roots=(root,),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_topology_replay=None,
        helicity_recurrence=None,
        helicity_materialization=None,
    )

    rewritten = assign_recursive_current_evaluation_reuse(dag, model)

    parent, proportional_parent, child, proportional_child = rewritten.interactions
    assert parent.evaluation_group_id == proportional_parent.evaluation_group_id
    assert child.evaluation_group_id == proportional_child.evaluation_group_id
    assert proportional_child.evaluation_factor == _complex_weight_mul(
        (2.0, 0.0),
        child.evaluation_factor,
    )
    # Roots consume separately materialized currents. Rewriting a shared child
    # kernel must therefore leave the physical amplitude contraction untouched.
    assert rewritten.amplitude_roots == dag.amplitude_roots
