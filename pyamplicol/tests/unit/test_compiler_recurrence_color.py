# SPDX-License-Identifier: 0BSD

from dataclasses import replace

import pytest

from pyamplicol.models.compiler_color_flow import (
    compile_lc_color_transition_terms,
)
from pyamplicol.models.contracts import CompiledOrientedKernel


def _kernel(structure: str) -> CompiledOrientedKernel:
    return CompiledOrientedKernel(
        kind=7,
        term_id=3,
        vertex="V_1",
        particles=("left", "right", "result"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="exact-color-source",
        color_expression="exact-color-expression",
        color_projection_structure=structure,
        color_projection_coefficient=(1.0, 0.0),
    )


@pytest.mark.parametrize(
    ("representations", "permutation", "component"),
    (
        ((3, 8, -3), (0, 1), "open-string"),
        ((8, 3, -3), (1, 0), "open-string"),
        ((8, -3, 3), (0, 1), "open-string"),
        ((-3, 8, 3), (1, 0), "open-string"),
        ((3, -3, 8), (0, 1), "adjoint-segment"),
        ((-3, 3, 8), (1, 0), "adjoint-segment"),
    ),
)
def test_fundamental_generator_terms_are_orientation_explicit(
    representations: tuple[int, int, int],
    permutation: tuple[int, int],
    component: str,
) -> None:
    terms = compile_lc_color_transition_terms(
        _kernel("fundamental-generator"),
        representations,
        proof_source="test-exact-projection",
    )
    assert len(terms) == 1
    assert terms[0].input_permutation == permutation
    assert terms[0].component_operation == "concatenate-join"
    assert terms[0].result_component_kind == component


def test_adjoint_structure_constant_is_an_exact_commutator() -> None:
    terms = compile_lc_color_transition_terms(
        _kernel("adjoint-structure-constant"),
        (8, 8, 8),
        proof_source="test-exact-projection",
    )
    assert tuple(term.input_permutation for term in terms) == ((0, 1), (1, 0))
    assert tuple(term.exact_factor_expression for term in terms) == ("1", "-1")
    assert len({term.proof_digest for term in terms}) == 2


def test_colored_literal_singlet_fails_without_contact_provenance() -> None:
    with pytest.raises(ValueError, match="requires explicit contact provenance"):
        compile_lc_color_transition_terms(
            replace(_kernel("singlet"), particles=("a", "b", "c")),
            (3, -3, 1),
            proof_source="test-exact-projection",
        )
