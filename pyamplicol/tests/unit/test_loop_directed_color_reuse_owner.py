# SPDX-License-Identifier: 0BSD
"""Actual prepared-owner checks; injection never regenerates process graphs."""

from copy import copy
from dataclasses import replace

import pytest

RETAINED_COLOR_REUSE = None


@pytest.fixture(scope="module")
def actual_reuse():
    if RETAINED_COLOR_REUSE is None:
        pytest.skip("requires retained same-process formal full/LC/NLC owners")
    return RETAINED_COLOR_REUSE


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_reprojected_bank_belongs_to_target_and_all_its_sources(actual_reuse, accuracy):
    control = actual_reuse
    target = control["targets"][accuracy]
    bank = control["prepared"][accuracy].bare.color_bank
    bank.require_owner(target)
    assert bank.plan is target.prepared
    assert bank.formal is target.bare.formal_color_metrics
    assert bank.projection.accuracy == accuracy
    assert bank.current_count == len(target.currents)
    assert bank.projection.families is control["bank"].projection.families
    assert (
        bank.projection.history_families is control["bank"].projection.history_families
    )
    assert all(
        item.bare.color_bank is bank for item in control["classes"][accuracy].values()
    )
    assert bank.plan is not control["bank"].plan
    assert bank.formal is not control["bank"].formal
    control["bank"].require_owner(control["reference"])


@pytest.mark.parametrize(
    "change", ("source", "captured_source", "process", "scalar", "formal", "bank")
)
def test_reuse_rejects_foreign_actual_owner_boundaries(actual_reuse, change):
    from pyamplicol.loops.directed_color_projection import reproject_directed_color_bank

    control = actual_reuse
    reference, bank = control["reference"], control["bank"]
    target = copy(control["targets"]["lc"])
    target.bare = copy(target.bare)
    if change == "source":
        target.source_color = None
    elif change == "captured_source":
        target._prepared_model = replace(
            target.prepared,
            metadata=tuple(
                (name, None if name == "source_color" else value)
                for name, value in target.prepared.metadata
            ),
        )
    elif change == "process":
        target.bare.process = control["other"].bare.process
    elif change == "scalar":
        target.bare.model_state = copy(target.bare.model_state)
        target.bare.model_state.plan = control["other"].bare.model_state.plan
    elif change == "formal":
        target.bare.formal_color_metrics = None
    elif change == "bank":
        bank = replace(bank, plan=target.prepared)
    with pytest.raises(ValueError, match=r"owner|process|source|projection"):
        reproject_directed_color_bank(reference, bank, target)
    control["bank"].require_owner(reference)
    control["prepared"]["lc"].bare.color_bank.require_owner(control["targets"]["lc"])
