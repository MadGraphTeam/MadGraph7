# SPDX-License-Identifier: 0BSD
"""A generation history is a prior, not a belief: with a history 0.01x,
0.1x, 10x or 100x off, the whole run's ETA follows this run's measured pace
once its first phases finish (predicted end within a factor 1.5 of the truth
from 25 % of the run on)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from pyamplicol.generation import progress as generation_progress
from pyamplicol.generation.progress import GenerationPhaseReporter
from pyamplicol.reporting import progress
from pyamplicol.reporting.eta import count_estimate


class _Recorder:
    """A sink keeping the task states as the display does."""

    def __init__(self) -> None:
        self.states: dict[str, progress._TaskState] = {}
        self.lock = threading.Lock()

    def emit(self, event) -> None:
        with self.lock:
            progress._track_event(self.states, event, time.monotonic())

    def total_eta(self, now: float) -> float | None:
        with self.lock:
            tasks = [task.copy() for task in self.states.values()]
        active = [task for task in tasks if task.finished_at is None]
        roots = [task for task in active if task.parent_task_id is None]
        if not roots:
            return None
        root = max(roots, key=lambda task: task.updated_at)
        children = [task for task in active if task.parent_task_id == root.task_id]
        phase = max(children, key=lambda task: task.updated_at) if children else root
        derived = progress._with_total_estimate(root, phase, now)
        value = derived.details.get("eta_seconds")
        return None if value is None else float(value)


def _run(sink) -> None:
    """A fake generation of ~2.4 s: two short steps, five steps, three items."""

    reporter = GenerationPhaseReporter(sink)
    reporter.start_generation(total=3, details={}, signature="stale-history")
    with reporter.phase("prepare", "Preparing", total=1):
        for key in ("a", "b"):
            with generation_progress.step(key, f"step {key}"):
                time.sleep(0.15)
    with reporter.phase("compile", "Compiling", total=1):
        for index in range(5):
            with generation_progress.step(f"s{index}", f"step {index}"):
                time.sleep(0.3)
    with reporter.phase("write", "Writing", total=3) as writing:
        for index in range(3):
            time.sleep(0.2)
            writing.update(index + 1)
    reporter.finish_generation()


@pytest.mark.parametrize("off", [0.01, 0.1, 10.0, 100.0])
def test_a_stale_history_gives_way_to_the_measured_run(
    off, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(tmp_path))
    _run(_Recorder())  # the true history
    path = tmp_path / "generation-history.json"
    data = json.loads(path.read_text())
    for record in data.values():
        record["steps"] = {key: value * off for key, value in record["steps"].items()}
    path.write_text(json.dumps(data))

    recorder = _Recorder()
    samples: list[tuple[float, float | None]] = []
    started = time.monotonic()
    done = threading.Event()

    def sample() -> None:
        while not done.wait(0.05):
            now = time.monotonic()
            samples.append((now - started, recorder.total_eta(now)))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    _run(recorder)
    done.set()
    sampler.join()
    total = time.monotonic() - started
    checked = [
        (elapsed, eta)
        for elapsed, eta in samples
        if 0.25 * total <= elapsed <= 0.95 * total
    ]
    assert checked
    for elapsed, eta in checked:
        # No ETA while work runs is "overdue" on the display: due now.
        predicted = elapsed + (eta or 0.0)
        assert total / 1.5 <= predicted <= 1.5 * total, (off, elapsed, eta, total)


def test_a_prior_far_from_the_measured_pace_is_dropped() -> None:
    marks = [(1.0, 1), (2.0, 2), (3.0, 3)]
    # The prior says 100 s per item, the run does 1 s per item.
    estimate = count_estimate(
        total=10,
        completed=3,
        started_at=0.0,
        marks=marks,
        now=3.0,
        prior_seconds=1000.0,
    )
    assert estimate.seconds == pytest.approx(7.0)
    assert estimate.estimated
    # A prior within a factor of a few is blended as before.
    blended = count_estimate(
        total=10, completed=3, started_at=0.0, marks=marks, now=3.0, prior_seconds=20.0
    )
    assert 7.0 < blended.seconds < 14.0 and not blended.estimated
