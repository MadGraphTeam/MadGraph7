# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.reporting import CallbackProgressSink, ProgressUpdate


def test_phase_advances_are_thread_safe_monotonic_and_exact() -> None:
    events: list[object] = []
    handle = PhaseHandle(
        "generation:test",
        CallbackProgressSink(events.append),
        total=200,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(handle.advance, message=str(index)) for index in range(200)
        ]
        for future in futures:
            future.result()

    updates = [event for event in events if isinstance(event, ProgressUpdate)]
    assert handle.completed == 200
    assert [event.completed for event in updates] == list(range(1, 201))
    assert {event.total for event in updates} == {200}
