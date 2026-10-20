# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol.api.results import BenchmarkComponentTiming, BenchmarkStatistics
from pyamplicol.config import R2Strategy
from pyamplicol.loops.strategy import (
    R2Branch,
    R2Capabilities,
    R2Comparison,
    R2Workload,
    select_r2_strategy,
)

pytestmark = pytest.mark.loop_fast

ARITHMETICS = frozenset({"double", "double-double", "arbitrary"})
MODEL = R2Branch(R2Strategy.MODEL, ARITHMETICS)
DIRECT = R2Branch(R2Strategy.DIRECT, ARITHMETICS)
BOTH = R2Capabilities((MODEL, DIRECT))
WORKLOAD = R2Workload("artifact", "jit", "cpu", "double", 53, "total", "parameters-0")
ERROR = BenchmarkStatistics(0.1, 0.05, 0.01)


def comparison(model_time=1.0, direct_time=2.0) -> R2Comparison:
    return R2Comparison(
        WORKLOAD,
        16,
        BenchmarkComponentTiming(model_time, ERROR, 8),
        BenchmarkComponentTiming(direct_time, ERROR, 8),
        ERROR,
    )


def select(capabilities=BOTH, workload=WORKLOAD, **kwargs):
    return select_r2_strategy("auto", capabilities, workload, batch_size=16, **kwargs)


def test_auto_chooses_direct_without_evidence_and_model_with_a_clear_win() -> None:
    assert select().effective is R2Strategy.DIRECT
    assert select(comparison=comparison()).effective is R2Strategy.MODEL
    assert select(comparison=comparison(1.9, 2.0)).effective is R2Strategy.DIRECT
    assert select(comparison=comparison(3.0, 2.0)).effective is R2Strategy.DIRECT


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_identity", "other"),
        ("backend", "cpp"),
        ("hardware", "other-cpu"),
        ("arithmetic", "arbitrary"),
        ("precision_bits", 256),
        ("observable_signature", "spin-correlated"),
        ("parameter_state", "parameters-1"),
        ("rescue_plan", (("arbitrary", 256),)),
        ("stability_policy", "relative-1e-10"),
    ],
)
def test_incompatible_or_stale_profile_restores_default(field, value) -> None:
    assert (
        select(
            workload=replace(WORKLOAD, **{field: value}), comparison=comparison()
        ).effective
        is R2Strategy.DIRECT
    )


def test_batch_sizes_are_not_extrapolated() -> None:
    assert (
        select_r2_strategy(
            "auto", BOTH, WORKLOAD, batch_size=128, comparison=comparison()
        ).effective
        is R2Strategy.DIRECT
    )


def test_only_complete_branch_is_selected_and_uv_is_always_required() -> None:
    assert select(R2Capabilities((DIRECT,))).effective is R2Strategy.DIRECT
    assert select(R2Capabilities((MODEL,))).effective is R2Strategy.MODEL
    chiral = replace(DIRECT, blockers=("uncertified dimensional gamma-five kernel",))
    assert select(R2Capabilities((MODEL, chiral))).effective is R2Strategy.MODEL
    with pytest.raises(ValueError, match="missing UV"):
        select(R2Capabilities((MODEL, DIRECT), ("missing UV",)))


def test_forcing_cannot_reenable_a_pruned_or_uncertified_branch() -> None:
    assert (
        select_r2_strategy("model", BOTH, WORKLOAD, batch_size=16).effective
        is R2Strategy.MODEL
    )
    with pytest.raises(ValueError, match="absent"):
        select_r2_strategy("model", R2Capabilities((DIRECT,)), WORKLOAD, batch_size=16)
    with pytest.raises(ValueError, match="gamma-five"):
        select_r2_strategy(
            "direct",
            R2Capabilities((replace(DIRECT, blockers=("gamma-five",)),)),
            WORKLOAD,
            batch_size=16,
        )


def test_rescue_capability_is_checked_before_selecting_the_call_strategy() -> None:
    limited = replace(DIRECT, arithmetics=frozenset({"double"}))
    capabilities = R2Capabilities((MODEL, limited))
    with_rescue = replace(WORKLOAD, rescue_plan=(("arbitrary", 256),))
    assert select(capabilities).effective is R2Strategy.DIRECT
    assert select(capabilities, workload=with_rescue).effective is R2Strategy.MODEL
    with pytest.raises(ValueError, match="missing arithmetic: arbitrary"):
        select_r2_strategy(
            "direct",
            capabilities,
            with_rescue,
            batch_size=16,
        )


def test_rescue_precision_change_invalidates_profile() -> None:
    original = replace(WORKLOAD, rescue_plan=(("arbitrary", 128),))
    profile = replace(comparison(), workload=original)
    assert select(workload=original, comparison=profile).effective is R2Strategy.MODEL
    changed = replace(original, rescue_plan=(("arbitrary", 256),))
    assert select(workload=changed, comparison=profile).effective is R2Strategy.DIRECT
