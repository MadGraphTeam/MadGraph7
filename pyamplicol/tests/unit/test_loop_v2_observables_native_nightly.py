# SPDX-License-Identifier: 0BSD
"""v2 observable controls in the precise carriers (WP2.5; nightly).

The 200-bit, DoubleFloat and 40-digit variants of the default tier's checks
in ``test_loop_v2_observables.py``: the asymmetric Ward identity (O10),
Recola's dipoles and spin source, native = Python, the stability estimates
and budgets, the resolved-sum identity of g g > g g, spin bilinearity at LC
and NLC, and the pure permutation alias d~ d > g g.
"""

from __future__ import annotations

import pytest

from .test_loop_v2_observables import (
    RESOLVED_CONTROLS,
    RESOLVED_SUM_DEFAULT,
    WARD_CONTROLS,
    assert_dipoles_match_recola,
    assert_estimates_bracket,
    assert_members_equal_direct,
    assert_native_equals_python,
    assert_native_estimates_follow_the_api,
    assert_resolved_entries_sum,
    assert_rotation_budgets_below_binary64,
    assert_spin_source_matches_recola,
    assert_spin_sources_bilinear,
    assert_ward_identity,
    observable_builder,
)

pytestmark = pytest.mark.loop_nightly


@pytest.fixture(scope="module")
def generate(loop_sm_compiled_model, loop_native):
    return observable_builder(loop_sm_compiled_model, loop_native)


@pytest.mark.parametrize("name", tuple(WARD_CONTROLS))
def test_asymmetric_ward_identity_at_200_bits(generate, name):
    """O10 at 200 bits (Python and native MPFR) and natively in
    DoubleFloat."""

    assert_ward_identity(generate, name, "arbitrary-200")


def test_directed_dipoles_match_recola_at_200_bits(generate, loop_reference):
    assert_dipoles_match_recola(generate, loop_reference, "arbitrary-200")


def test_singleton_spin_source_matches_recola_at_200_bits(generate, loop_reference):
    assert_spin_source_matches_recola(generate, loop_reference, "arbitrary-200")


@pytest.mark.parametrize("carrier", ("double-double", "arbitrary-200"))
def test_native_observables_equal_python_precisely(generate, carrier):
    assert_native_equals_python(generate, carrier)


def test_observable_stability_estimates_bracket_the_true_error_at_40_digits(
    generate,
):
    assert_estimates_bracket(generate, "arbitrary-40")


@pytest.mark.parametrize("carrier", ("double-double", "arbitrary-40"))
def test_native_observable_estimates_follow_the_stability_api_precisely(
    generate, carrier
):
    assert_native_estimates_follow_the_api(generate, carrier)


def test_rotation_budgets_resolve_deviations_below_binary64_at_40_digits(generate):
    assert_rotation_budgets_below_binary64(generate, "arbitrary-40")


@pytest.mark.parametrize(
    "control",
    sorted({control for control, _ in RESOLVED_SUM_DEFAULT} & set(RESOLVED_CONTROLS)),
)
def test_resolved_entries_sum_to_the_inclusive_result_at_200_bits(generate, control):
    assert_resolved_entries_sum(generate, control, "arbitrary-200")


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_spin_sources_are_bilinear_at_lc_and_nlc(generate, accuracy):
    assert_spin_sources_bilinear(generate, accuracy)


def test_pure_alias_equals_direct_generation(generate):
    assert_members_equal_direct(generate, (("d~ d > g g", True),))
