# SPDX-License-Identifier: 0BSD
"""Select one complete rational-completion branch at evaluation-call entry."""

from __future__ import annotations

from dataclasses import dataclass

from pyamplicol.api.results import BenchmarkComponentTiming, BenchmarkStatistics
from pyamplicol.config import R2Strategy

_ARITHMETICS = frozenset({"double", "double-double", "arbitrary"})


@dataclass(frozen=True, slots=True)
class R2Branch:
    strategy: R2Strategy
    arithmetics: frozenset[str]
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        strategy = R2Strategy(self.strategy)
        if strategy is R2Strategy.AUTO:
            raise ValueError("an artifact branch must be model or direct")
        arithmetics = frozenset(self.arithmetics)
        if not arithmetics <= _ARITHMETICS:
            raise ValueError("unknown branch arithmetic")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "arithmetics", arithmetics)
        object.__setattr__(self, "blockers", tuple(self.blockers))


@dataclass(frozen=True, slots=True)
class R2Capabilities:
    """Only certified branches present in the generated artifact may be selected.

    Common blockers include missing UV or scheme-restoring terms. Raw UFO presence
    alone cannot produce a certified branch.
    """

    branches: tuple[R2Branch, ...]
    common_blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        branches = tuple(self.branches)
        if len({branch.strategy for branch in branches}) != len(branches):
            raise ValueError("duplicate rational-completion branches")
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "common_blockers", tuple(self.common_blockers))


@dataclass(frozen=True, slots=True)
class R2Workload:
    """References existing identities; never hashes event momenta or sources.

    The observable signature describes selectors and source *classes*, not their
    individual numerical vectors. Parameter state refers to the runtime's global
    parameter identity; updates therefore invalidate hints without touching kernels.
    """

    artifact_identity: str
    backend: str
    hardware: str
    arithmetic: str
    precision_bits: int
    observable_signature: str
    parameter_state: str
    rescue_plan: tuple[tuple[str, int], ...] = ()
    stability_policy: str = "none"

    def __post_init__(self) -> None:
        if self.arithmetic not in _ARITHMETICS:
            raise ValueError("unknown workload arithmetic")
        if (
            isinstance(self.precision_bits, bool)
            or not isinstance(self.precision_bits, int)
            or self.precision_bits < 2
        ):
            raise ValueError("precision_bits must be an integer of at least two")
        rescue = tuple(tuple(attempt) for attempt in self.rescue_plan)
        for arithmetic, bits in rescue:
            if arithmetic not in _ARITHMETICS or (
                isinstance(bits, bool) or not isinstance(bits, int) or bits < 2
            ):
                raise ValueError("invalid rescue arithmetic or precision")
        object.__setattr__(self, "rescue_plan", rescue)


@dataclass(frozen=True, slots=True)
class R2Comparison:
    """Full independent pipeline timings from paired calibrated profile blocks.

    This is an optional profiling result, not an artifact manifest. The paired
    uncertainty is measured for the difference (direct minus model) by the existing
    profiler. Shared-event intermediate reuse cannot be used for these timings.
    """

    workload: R2Workload
    batch_size: int
    model: BenchmarkComponentTiming
    direct: BenchmarkComponentTiming
    paired_uncertainty: BenchmarkStatistics

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("profile batch_size must be a measured positive size")
        if self.model.sample_count != self.direct.sample_count:
            raise ValueError("R2 timing comparison requires paired blocks")

    def demonstrates_model_win(self, workload: R2Workload, batch_size: int) -> bool:
        if self.workload != workload or batch_size != self.batch_size:
            return False
        if self.model.sample_count < 2:
            return False
        if (
            min(self.model.mean_seconds_per_point, self.direct.mean_seconds_per_point)
            <= 0
        ):
            return False
        difference = (
            self.direct.mean_seconds_per_point - self.model.mean_seconds_per_point
        )
        return difference > 3 * self.paired_uncertainty.standard_error


@dataclass(frozen=True, slots=True)
class R2Selection:
    requested: R2Strategy
    effective: R2Strategy
    reason: str


def select_r2_strategy(
    requested: R2Strategy,
    capabilities: R2Capabilities,
    workload: R2Workload,
    *,
    batch_size: int,
    comparison: R2Comparison | None = None,
) -> R2Selection:
    """Return the immutable choice to reuse for this entire call and its rescue.

    This function performs no timing and examines no event values. A strategy's
    eligibility includes every configured rescue arithmetic before any work starts.
    """
    requested = R2Strategy(requested)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 0
    ):
        raise ValueError("batch_size must be a nonnegative integer")
    required = {workload.arithmetic, *(attempt[0] for attempt in workload.rescue_plan)}
    available: set[R2Strategy] = set()
    reasons: dict[R2Strategy, str] = {}
    for branch in capabilities.branches:
        blockers = (*capabilities.common_blockers, *branch.blockers)
        missing = sorted(required - branch.arithmetics)
        if missing:
            blockers += ("missing arithmetic: " + ", ".join(missing),)
        if blockers:
            reasons[branch.strategy] = "; ".join(blockers)
        else:
            available.add(branch.strategy)
    if requested is not R2Strategy.AUTO:
        if requested not in available:
            reason = reasons.get(requested, "branch is absent from this artifact")
            raise ValueError(
                f"R2 strategy {requested.value!r} is unavailable: {reason}"
            )
        return R2Selection(requested, requested, "explicit strategy")
    if not available:
        detail = "; ".join(
            f"{strategy.value}: {reasons.get(strategy, 'branch absent')}"
            for strategy in (R2Strategy.MODEL, R2Strategy.DIRECT)
        )
        raise ValueError(f"no complete eligible R2 strategy: {detail}")
    if len(available) == 1:
        return R2Selection(
            requested, next(iter(available)), "only complete eligible strategy"
        )
    if comparison is not None and comparison.demonstrates_model_win(
        workload, batch_size
    ):
        return R2Selection(
            requested,
            R2Strategy.MODEL,
            "matching profile demonstrates a model-rule win",
        )
    return R2Selection(
        requested,
        R2Strategy.DIRECT,
        "direct default; no conclusive matching model-rule win",
    )
