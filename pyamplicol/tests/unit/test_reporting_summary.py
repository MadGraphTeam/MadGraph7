# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
from dataclasses import dataclass

from pyamplicol.cli.main import write_result
from pyamplicol.reporting import render_summary


@dataclass(frozen=True)
class _Result:
    status: str
    generated_processes: int
    adjustments: tuple[str, ...]


def test_human_structured_results_use_aligned_prettytable() -> None:
    rendered = render_summary(_Result("complete", 2, ()), color=False)
    assert rendered is not None
    assert "field" in rendered
    assert "generated processes" in rendered
    assert "complete" in rendered


def test_json_result_never_contains_table_or_color_sequences() -> None:
    stream = io.StringIO()
    write_result(_Result("complete", 2, ()), format="json", stream=stream, color=True)
    assert json.loads(stream.getvalue()) == {
        "adjustments": [],
        "generated_processes": 2,
        "status": "complete",
    }
    assert "\x1b[" not in stream.getvalue()
