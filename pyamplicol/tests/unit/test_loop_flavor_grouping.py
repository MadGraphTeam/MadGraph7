# SPDX-License-Identifier: 0BSD
"""Exact loop-flavour grouping helpers: species bijections and basis strength."""

from dataclasses import replace

import pytest

from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember
from pyamplicol.loops.flavor_mappings import identity_species_candidate

pytestmark = pytest.mark.loop_fast


def test_identity_species_candidates_preserve_signed_bijections_without_caps():
    assert identity_species_candidate((2, -2, 1, -1), (2, -2, 3, -3)) == {
        1: 3,
        -1: -3,
        3: 1,
        -3: -1,
    }
    # Three-cycle completion, not a hard-coded pairwise flavour swap.
    assert identity_species_candidate((1, 3), (3, 5)) == {
        1: 3,
        -1: -3,
        3: 5,
        -3: -5,
        5: 1,
        -5: -1,
    }
    assert identity_species_candidate((1, 1), (1, 3)) is None
    assert identity_species_candidate((1, -1), (3, -5)) is None
    assert identity_species_candidate((21,), (21,)) == {}


def test_exact_group_basis_is_weaker_than_identity_and_stronger_than_numerical():
    item = GroupMember("a", "a a > b b", (1, -1, 3, -3), 2, permutation=(0, 1, 2, 3))
    exact = replace(item, basis=GroupingBasis.EXACT_LOOP_FLAVOR)
    numerical = replace(item, basis=GroupingBasis.NUMERICAL)
    assert item.through(exact).basis == exact.basis
    assert exact.through(item).basis == exact.basis
    assert exact.through(numerical).basis == numerical.basis
