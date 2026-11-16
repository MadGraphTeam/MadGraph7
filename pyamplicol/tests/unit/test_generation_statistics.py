# SPDX-License-Identifier: 0BSD
"""The generation statistics: their measurement, their record in the artifact
metadata and their rows in the `generate` and `inspect` tables and JSON."""

from __future__ import annotations

import io
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pyamplicol.api import GenerationResult, GenerationStatistics, ProcessSet
from pyamplicol.api.requests import ProcessRequest
from pyamplicol.artifacts import inspect_artifact, load_manifest
from pyamplicol.artifacts.manifest import MANIFEST_NAME
from pyamplicol.cli.main import _json_value
from pyamplicol.generation.progress import GenerationPhaseReporter, record_phase
from pyamplicol.generation.statistics import (
    measure_generation,
    record_generation_statistics,
)
from pyamplicol.reporting import render_summary
from pyamplicol.reporting.descriptions import field_descriptions
from pyamplicol.reporting.resources import peak_memory

ROOT = Path(__file__).resolve().parents[2]
# Written before pyAmpliCol recorded generation statistics.
OLD_ARTIFACT = ROOT / "src/pyamplicol/assets/selftest/portable-64le/artifact"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINUX = platform.system() == "Linux"


def _statistics(**fields: object) -> GenerationStatistics:
    values: dict[str, object] = {
        "wall_seconds": 34.2,
        "phases": (
            ("setup", 1.4),
            ("loop-model", 19.17),
            ("loop-compile", 13.85),
        ),
        "unattributed_seconds": 34.2 - 1.4 - 19.17 - 13.85,
        "peak_rss_bytes": 1288490188,
        "peak_rss_bound": "exact",
        "peak_rss_method": "VmHWM after clear_refs; descendants every 1 s",
        "process_peak_rss_bytes": 1288490188,
        "child_processes": False,
        "child_peak_rss_bytes": None,
    }
    values.update(fields)
    return GenerationStatistics(**values)  # type: ignore[arg-type]


def _row(rendered: str, field: str) -> list[str]:
    """The value lines of ``field``'s row (its continuation lines too)."""

    lines = _ANSI.sub("", rendered).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {field} "))
    values = [lines[start].split("|")[2].strip()]
    for line in lines[start + 1 :]:
        cells = line.split("|")
        if len(cells) < 3 or cells[1].strip():
            break
        values.append(cells[2].strip())
    return values


def test_an_old_artifact_inspects_with_a_muted_dash(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    shutil.copytree(OLD_ARTIFACT, root)
    inspection = inspect_artifact(root)

    assert inspection.generation_statistics is None
    rendered = render_summary(inspection, color=False)
    assert rendered is not None
    assert _row(rendered, "generation time") == ["-"]
    assert _row(rendered, "peak RAM") == ["-"]
    coloured = render_summary(inspection, color=True)
    assert coloured is not None and "\x1b[90m-" in coloured
    assert _json_value(inspection)["generation_statistics"] is None


def test_the_record_keeps_the_artifact_identity_and_inspects(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    shutil.copytree(OLD_ARTIFACT, root)
    before = load_manifest(root, verify_payloads=True)
    statistics = _statistics()

    assert record_generation_statistics(root, statistics)

    after = load_manifest(root, verify_payloads=True)
    assert after.artifact_id == before.artifact_id
    stored = json.loads((root / MANIFEST_NAME).read_text())["extensions"]
    assert stored["generation"]["statistics"] == statistics.to_json()
    inspection = inspect_artifact(root)
    assert inspection.generation_statistics == statistics
    rendered = render_summary(inspection, color=False)
    assert rendered is not None
    assert _row(rendered, "generation time")[0].startswith("34.2 s (setup 1.4 s, ")
    assert _row(rendered, "peak RAM") == ["1.2 GiB"]
    payload = _json_value(inspection)["generation_statistics"]
    assert payload["wall_seconds"] == 34.2
    assert payload["phases"][1] == {"phase": "loop-model", "seconds": 19.17}
    assert payload["peak_rss_bytes"] == 1288490188


def test_a_record_of_another_schema_reads_as_not_recorded() -> None:
    assert GenerationStatistics.from_json(None) is None
    assert GenerationStatistics.from_json({"schema_version": 2}) is None
    assert GenerationStatistics.from_json({"schema_version": 1}) is None
    statistics = _statistics()
    assert GenerationStatistics.from_json(statistics.to_json()) == statistics


def test_generate_rows_fit_the_table_and_the_json_names_the_units() -> None:
    phases = tuple((f"phase-number-{index}", 1.5 * index) for index in range(1, 13))
    result = GenerationResult(
        output=Path("artifact"),
        processes=ProcessSet((ProcessRequest.parse("d d~ > z"),)),
        mode="error",
        statistics=_statistics(
            phases=phases, wall_seconds=125.0, unattributed_seconds=8.0
        ),
    )

    rendered = render_summary(result, color=False)
    assert rendered is not None
    time_lines = _row(rendered, "generation time")
    assert time_lines[0].startswith("2m05s (phase-number-1 1.5 s, ")
    assert time_lines[-1].endswith("unattributed 8 s)")
    assert len(time_lines) > 1
    assert all(len(line) <= 88 for line in time_lines)
    # whole "phase time" items per line
    assert all(re.fullmatch(r"(\S+ \S+ \S+,? ?)+|.*\)", line) for line in time_lines)
    assert _row(rendered, "peak RAM") == ["1.2 GiB"]
    coloured = render_summary(result, color=True)
    assert coloured is not None
    assert "\x1b[1m2m05s" in coloured and "\x1b[90m (phase-number-1" in coloured

    stream = io.StringIO()
    from pyamplicol.cli.main import write_result

    write_result(result, format="json", stream=stream, color=True)
    payload = json.loads(stream.getvalue())["statistics"]
    assert payload["wall_seconds"] == 125.0
    assert payload["phases"][1] == {"phase": "phase-number-2", "seconds": 3.0}
    assert isinstance(payload["peak_rss_bytes"], int)
    assert payload["peak_rss_bound"] == "exact"
    assert payload["peak_rss_method"].startswith("VmHWM")
    assert "\x1b[" not in stream.getvalue()


def test_started_processes_make_the_peak_a_lower_bound() -> None:
    result = GenerationResult(
        output=Path("artifact"),
        processes=ProcessSet((ProcessRequest.parse("d d~ > z"),)),
        mode="error",
        statistics=_statistics(
            peak_rss_bound="lower bound",
            child_processes=True,
            child_peak_rss_bytes=300 * 1024**2,
        ),
    )
    rendered = render_summary(result, color=False)
    assert rendered is not None
    assert _row(rendered, "peak RAM") == [
        ">= 1.2 GiB (process 1.2 GiB, largest started process 300 MiB)"
    ]
    described = field_descriptions(result)
    assert "generation time" in described and "'>= x'" in described


def test_the_phases_and_the_rest_add_up_to_the_wall_time() -> None:
    with measure_generation() as measured:
        record_phase("setup", 0.0)
        reporter = GenerationPhaseReporter(None)
        with reporter.phase("first", "first"):
            time.sleep(0.05)
            # a nested generation's phase is part of the open one
            with GenerationPhaseReporter(None).phase("nested", "nested"):
                time.sleep(0.01)
        time.sleep(0.02)
        reporter.record("between", 0.01)
        with reporter.phase("second", "second"):
            time.sleep(0.03)
    statistics = measured.statistics

    assert statistics is not None
    assert [name for name, _ in statistics.phases] == [
        "setup",
        "first",
        "between",
        "second",
    ]
    total = sum(seconds for _, seconds in statistics.phases)
    assert total + statistics.unattributed_seconds == pytest.approx(
        statistics.wall_seconds
    )
    assert statistics.unattributed_seconds >= 0.01
    assert statistics.peak_rss_bytes is not None or not _LINUX
    if _LINUX:
        assert statistics.child_processes is False
        assert statistics.peak_rss_bound == "exact"
        assert statistics.peak_rss_bytes == statistics.process_peak_rss_bytes


@pytest.mark.skipif(not _LINUX, reason="RUSAGE_CHILDREN is read on Linux only")
def test_a_started_process_is_counted() -> None:
    size = 160 * 1024**2
    with measure_generation(interval_seconds=0.05) as measured:
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"import time; b = bytearray({size}); time.sleep(0.3)",
            ],
            check=True,
        )
    statistics = measured.statistics

    assert statistics is not None
    assert statistics.child_processes is True
    assert statistics.peak_rss_bound == "lower bound"
    assert statistics.child_peak_rss_bytes is not None
    assert statistics.child_peak_rss_bytes >= size
    assert statistics.peak_rss_bytes is not None
    assert statistics.peak_rss_bytes >= statistics.child_peak_rss_bytes


@pytest.mark.skipif(not _LINUX, reason="the resettable mark is Linux's")
def test_a_nested_region_keeps_the_outer_peak() -> None:
    import mmap

    size = 128 * 1024**2
    with peak_memory() as outer:
        block = mmap.mmap(-1, size)
        for offset in range(0, size, mmap.PAGESIZE):
            block[offset] = 1
        block.close()
        with peak_memory() as inner:
            pass
    assert outer.result is not None and inner.result is not None
    assert outer.result.scope == "profiled region"
    assert outer.result.peak_rss_bytes is not None
    assert outer.result.start_rss_bytes is not None
    # the inner region did not reset the mark the outer one reads
    assert outer.result.peak_rss_bytes - outer.result.start_rss_bytes >= size // 2
    assert inner.result.method == "VmHWM"


def test_the_inspect_descriptions_cover_the_artifact_table(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    shutil.copytree(OLD_ARTIFACT, root)
    described = field_descriptions(inspect_artifact(root))
    for field in ("created", "generation time", "peak RAM", "integrity"):
        assert field in described
