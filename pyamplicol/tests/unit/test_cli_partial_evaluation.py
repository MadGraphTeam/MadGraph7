# SPDX-License-Identifier: 0BSD
"""``evaluate`` over successive calls: the concatenated result is the one-call
result, and a Ctrl-C keeps the tables of the points done (m-lit ux-cli C9)."""

import io
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.api.errors import EvaluationError
from pyamplicol.api.virtual_results import (
    VirtualEvaluation,
    VirtualFailureDiagnostic,
    VirtualLaneDiagnostic,
)
from pyamplicol.cli import partial_evaluation as pe
from pyamplicol.cli.main import write_result
from pyamplicol.config import R2Strategy
from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
from pyamplicol.loops.strategy import R2Selection
from pyamplicol.reporting import CallbackProgressSink
from pyamplicol.reporting.progress import NullProgressSink, ProgressUpdate

pytestmark = pytest.mark.loop_fast

_QCD2, _QCD4 = (("QCD", 2),), (("QCD", 4),)


def _lane(point: int, value: int) -> VirtualLaneDiagnostic:
    """A success lane for a positive value; a failed one (no value) whose
    failure numbers its point within the call otherwise."""

    if value > 0:
        return VirtualLaneDiagnostic("success", "double", 53, True)
    return VirtualLaneDiagnostic(
        "failed",
        "double",
        53,
        False,
        failures=(VirtualFailureDiagnostic(point, 0, 0, 0, "reduction", "x"),),
    )


def _virtual(values: list[int]) -> VirtualEvaluation:
    """The one-call result over ``values`` (one point each, point ids from
    0; a non-positive value is a failed point)."""

    return VirtualEvaluation(
        tuple((Decimal(v),) if v > 0 else None for v in values),
        tuple(
            ((Decimal(v), Decimal(-v), Decimal(0)),) if v > 0 else None for v in values
        ),
        (_QCD4,),
        (VirtualOrderSector((("QCD", 6),), (BornLoopPair(_QCD2, _QCD4),)),),
        tuple(_lane(index, v) for index, v in enumerate(values)),
        R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "only complete strategy"),
    )


def test_concatenation_is_the_one_call_result_with_shifted_point_ids():
    whole = _virtual([1, -2, 3, 4, -5])
    parts = [_virtual([1]), _virtual([-2, 3]), _virtual([4, -5])]
    assert pe.concatenate(parts, [0, 1, 3]) == whole
    assert pe.concatenate([(1, 2), (3,)], [0, 2]) == (1, 2, 3)


def test_concatenation_refuses_differing_common_fields():
    other = replace(
        _virtual([2]), r2=R2Selection(R2Strategy.AUTO, R2Strategy.MODEL, "y")
    )
    with pytest.raises(ValueError, match="r2"):
        pe.concatenate([_virtual([1]), other], [0, 1])


class _Clock:
    """A clock the fake operation advances by a call's cost: a fixed cost
    plus a cost per point (seconds)."""

    def __init__(self, fixed: float = 0.13, per_point: float = 0.02) -> None:
        self.now, self.fixed, self.per_point = 0.0, fixed, per_point

    def perf_counter(self) -> float:
        return self.now


def _operation(
    calls: list[int],
    clock: _Clock,
    interrupt_at: int | None = None,
    failing: int | None = None,
):
    def operation(momenta, *, mu_r2, scale):
        calls.append(len(momenta))
        clock.now += clock.fixed + clock.per_point * len(momenta)
        if interrupt_at is not None and len(calls) == interrupt_at:
            raise KeyboardInterrupt
        assert len(mu_r2) == len(momenta) and scale == 7
        for index, momentum in enumerate(momenta):
            if momentum == failing:
                # The runtime numbers a failing point within its call.
                raise EvaluationError(f"failed at point {index}")
        return tuple(m * 10 for m in momenta)

    return operation


def _run(monkeypatch, total: int, clock: _Clock, **kwargs):
    monkeypatch.setattr(pe, "time", clock)
    calls: list[int] = []
    events: list[object] = []
    result = pe.evaluate_in_chunks(
        CallbackProgressSink(events.append),
        _operation(calls, clock, **kwargs),
        list(range(total)),
        {"mu_r2": list(range(total))},
        {"scale": 7},
    )
    return result, calls, events


def test_chunks_cover_every_point_once_with_progress(monkeypatch):
    result, calls, events = _run(monkeypatch, 1000, _Clock())
    assert result == tuple(m * 10 for m in range(1000))
    # One point, eight measured warm, calls growing at most fourfold up to
    # (fixed / overhead - fixed) / per_point = 210 points, and the rest in
    # one call once it fits in one and a half such calls.
    assert calls == [1, 8, 32, 128, 210, 210, 210, 201]
    updates = [e.completed for e in events if isinstance(e, ProgressUpdate)]
    assert updates == [1, 9, 41, 169, 379, 589, 799, 1000]


def test_a_short_evaluation_takes_few_calls(monkeypatch):
    # The fixed cost of the extra calls stays a small share.
    _, calls, _ = _run(monkeypatch, 20, _Clock())
    assert calls == [1, 8, 11]
    _, calls, _ = _run(monkeypatch, 1, _Clock())
    assert calls == [1]


def test_a_cold_first_call_never_sends_the_rest_in_one_call(monkeypatch):
    # The first call can pay a first-call cost and be slower than the second
    # call with more points (u u~ > d d~: 0.21 s for 1 point, 0.11 s for 8).
    # The cold first call is left out of the cost fit: a falling slope
    # would otherwise read as no cost per point.
    assert pe._next_size([(1, 0.30), (8, 0.20)], 19991) == 32
    assert pe._next_size([(1, 0.21), (8, 0.11)], 2991) == 32
    # A flat warm fit (8 and 32 points equally fast) bounds the next call
    # by the growth factor, never the 19960 points left.
    assert pe._next_size([(1, 0.3), (8, 0.11), (32, 0.11)], 19960) == 128

    clock = _Clock(fixed=0.09, per_point=0.0027)
    monkeypatch.setattr(pe, "time", clock)
    calls: list[int] = []
    operation = _operation(calls, clock)

    def cold(momenta, **kwargs):
        if not calls:
            clock.now += 0.2
        return operation(momenta, **kwargs)

    result = pe.evaluate_in_chunks(
        NullProgressSink(),
        cold,
        list(range(20000)),
        {"mu_r2": list(range(20000))},
        {"scale": 7},
    )
    assert result == tuple(m * 10 for m in range(20000))
    assert calls[:5] == [1, 6, 24, 96, 384]
    # No call longer than one and a half target times after the first two.
    longest = max(0.09 + 0.0027 * n for n in calls[2:])
    assert longest <= 1.5 * max(2.0, 0.09 / 0.03) + 0.1
    # The fixed cost of the extra calls stays a few per cent.
    assert len(calls) * 0.09 <= 0.05 * (0.09 + 0.0027 * 20000)


def test_the_chunk_policy_is_overridable(monkeypatch):
    policy = pe.ChunkPolicy(seconds=0.5, overhead=0.5, probe_points=2, growth=2.0)
    assert pe._next_size([(1, 0.1)], 100, policy) == 2
    assert pe._next_size([(1, 0.1), (2, 0.2)], 100, policy) == 4


def test_slow_points_go_one_per_call_after_the_second_call(monkeypatch):
    _, calls, _ = _run(monkeypatch, 6, _Clock(fixed=0.0, per_point=30.0))
    assert calls == [1, 2, 1, 1, 1]


def test_a_failure_is_the_one_call_failure(monkeypatch):
    # Point 12 fails in the third call; the error names it by its index in
    # all the points, as one call does.
    with pytest.raises(EvaluationError, match=r"failed at point 12$"):
        _run(monkeypatch, 20, _Clock(), failing=12)
    calls: list[int] = []
    clock = _Clock()
    monkeypatch.setattr(pe, "time", clock)
    with pytest.raises(EvaluationError, match=r"failed at point 12$"):
        pe.evaluate_in_chunks(
            NullProgressSink(),
            _operation(calls, clock, failing=12),
            list(range(20)),
            {"mu_r2": list(range(20))},
            {"scale": 7},
        )
    assert calls == [1, 8, 11, 20]


def test_interrupt_keeps_the_points_done(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(pe, "time", clock)
    calls: list[int] = []
    partial = pe.evaluate_in_chunks(
        NullProgressSink(),
        _operation(calls, clock, interrupt_at=3),
        list(range(500)),
        {"mu_r2": list(range(500))},
        {"scale": 7},
    )
    assert isinstance(partial, pe.PartialEvaluation)
    assert (partial.result, partial.requested, partial.measured) == (
        tuple(m * 10 for m in range(9)),
        500,
        9,
    )
    machine, human = io.StringIO(), io.StringIO()
    tree = pe.PartialEvaluation((0, 10), 5, 2)
    write_result(tree, format="json", stream=machine)
    # A tree evaluation's list keeps its type; the notice goes to stderr.
    assert json.loads(machine.getvalue()) == [0, 10]
    virtual = pe.PartialEvaluation(_virtual([1, 2]), 9, 2)
    write_result(virtual, format="human", stream=human)
    machine = io.StringIO()
    write_result(virtual, format="json", stream=machine)
    notice = (
        "interrupted: 2 of 9 requested points evaluated; the tables hold those points"
    )
    assert human.getvalue().rstrip().endswith(notice)
    payload = json.loads(machine.getvalue())
    assert payload["shape"][0] == 2 and payload["measured_points"] == 2


def test_interrupt_before_any_point_propagates(monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        pe.evaluate_in_chunks(
            NullProgressSink(),
            _operation([], _Clock(), interrupt_at=1),
            list(range(3)),
            {"mu_r2": list(range(3))},
            {"scale": 7},
        )


def test_field_descriptions_name_the_columns_of_every_command():
    from pyamplicol.diagnostics import DiagnosticCheck, DiagnosticReport
    from pyamplicol.reporting.descriptions import field_descriptions

    tree = " ".join(field_descriptions((1.0, -2.0)).split())
    assert "Summed Evaluation" in tree and "matrix element -" in tree
    report = DiagnosticReport("1.0.3", (DiagnosticCheck("python", "pass", "3.11"),))
    text = field_descriptions(report)
    assert "Checks" in text and "status" in text and "package version" in text
    whole = " ".join(field_descriptions(_virtual([1])).split())
    assert "single pole [-1] - the 1/epsilon coefficient of the renormalised" in whole
    nlc = field_descriptions(replace(_virtual([1]), color_accuracy="nlc"))
    assert "Born-loop interference, next-to-leading colour" in nlc


def test_only_a_live_display_evaluates_in_successive_calls():
    from pyamplicol.reporting.progress import TtyProgressSink

    class _Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert TtyProgressSink(_Terminal()).live
    assert not TtyProgressSink(io.StringIO()).live
    assert not getattr(NullProgressSink(), "live", False)
