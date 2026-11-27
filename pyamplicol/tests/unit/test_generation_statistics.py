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
from dataclasses import replace
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


@pytest.fixture
def owned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as pyAmpliCol's own process (its command line, a worker), where
    the peak is measured from the reset high-water mark."""

    from pyamplicol.reporting import resources

    monkeypatch.setattr(resources, "_OWNS_PROCESS", True)


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
            peak_rss_bytes=1536 * 1024**2,
            peak_rss_bound="lower bound",
            child_processes=True,
            child_peak_rss_bytes=1536 * 1024**2,
        ),
    )
    rendered = render_summary(result, color=False)
    assert rendered is not None
    assert _row(rendered, "peak RAM") == [
        ">= 1.5 GiB (process 1.2 GiB, largest started process 1.5 GiB)"
    ]
    own = render_summary(
        replace(
            result,
            statistics=_statistics(peak_rss_bound="lower bound", child_processes=True),
        ),
        color=False,
    )
    assert own is not None
    assert _row(own, "peak RAM") == [">= 1.2 GiB"]
    described = field_descriptions(result)
    assert "generation time" in described and "'>= x'" in described


def test_the_phases_and_the_rest_add_up_to_the_wall_time(owned: None) -> None:
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
def test_a_started_process_is_counted(owned: None) -> None:
    import resource

    from pyamplicol.reporting import resources

    # Above this process's own size: the kernel credits the started process
    # with the resident size of the image it is started from.
    current = resources._proc_status_kib("VmRSS") or 0
    size = current + 192 * 1024**2
    # The largest peak of the processes earlier tests of this process
    # started: RUSAGE_CHILDREN keeps it, a smaller one cannot raise it.
    earlier = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
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
    if "not attributable" in statistics.peak_rss_method or earlier >= size:
        # a process of an earlier test ended inside the region, or a larger
        # one ended before it
        return
    assert statistics.child_peak_rss_bytes is not None
    assert statistics.child_peak_rss_bytes >= size
    assert statistics.process_peak_rss_bytes is not None
    assert statistics.child_peak_rss_bytes > statistics.process_peak_rss_bytes
    assert statistics.peak_rss_bytes is not None
    assert statistics.peak_rss_bytes >= statistics.child_peak_rss_bytes


@pytest.mark.skipif(not _LINUX, reason="the processes are read from /proc")
def test_processes_running_before_the_region_are_not_the_regions(
    owned: None,
) -> None:
    """A process the caller started before (a pool, a helper, as the
    resource tracker a spawned pool of an earlier test leaves) is not one
    the generation started: neither while it runs nor when it ends inside
    the region (RUSAGE_CHILDREN cannot tell it from the region's own)."""

    sleep = [sys.executable, "-c", "import time; time.sleep(30)"]
    staying, ending = subprocess.Popen(sleep), subprocess.Popen(sleep)
    try:
        time.sleep(0.2)
        with measure_generation(interval_seconds=0.05) as measured:
            time.sleep(0.15)
            ending.terminate()
            ending.wait()
            time.sleep(0.1)
    finally:
        for process in (staying, ending):
            process.kill()
            process.wait()
    statistics = measured.statistics

    assert statistics is not None
    assert statistics.child_processes is False
    assert statistics.peak_rss_bound == "exact"
    assert "RUSAGE_CHILDREN not attributable" in statistics.peak_rss_method


@pytest.mark.skipif(not _LINUX, reason="the resettable mark is Linux's")
def test_a_nested_region_keeps_the_outer_peak(owned: None, monkeypatch) -> None:
    """Only the outermost region resets the kernel's high-water mark: an
    inner one reads it as it is (a reset would lose the outer region's
    peak). Checked on the reset itself: a peak read from memory released
    before the read depends on the kernel's per-CPU batched counters."""

    from pyamplicol.reporting import resources

    resets: list[str] = []

    class ClearRefs:
        def write_text(self, text: str) -> int:
            resets.append(text)
            return len(text)

    monkeypatch.setattr(resources, "_PROC_CLEAR_REFS", ClearRefs())
    with peak_memory() as outer:
        with peak_memory() as inner:
            pass
        with peak_memory() as second:
            pass
    with peak_memory() as after:
        pass

    assert resets == ["5", "5"]  # the outer region and the one after it
    assert outer.result is not None and outer.result.scope == "profiled region"
    assert inner.result is not None and inner.result.method == "VmHWM"
    assert second.result is not None and second.result.method == "VmHWM"
    assert after.result is not None
    assert after.result.method == "VmHWM after clear_refs"


def _callers_peak() -> tuple[int | None, int | None]:
    from pyamplicol.reporting import resources

    return resources._proc_status_kib("VmHWM"), resources._rusage_peak_bytes()


@pytest.mark.skipif(not _LINUX, reason="the resettable mark is Linux's")
def test_a_library_call_keeps_the_callers_peak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A program that uses pyAmpliCol as a library keeps its own peak RSS
    (VmHWM, ru_maxrss: what its parent's wait4 and `time -v` read): the
    process-wide reset is only done where pyAmpliCol owns the process. The
    generation's peak is then the lifetime peak: exact when the call raised
    it, else an upper bound."""

    import mmap

    from pyamplicol.api import Generator
    from pyamplicol.api import services as service_module
    from pyamplicol.config import GenerationConfig
    from pyamplicol.reporting import resources

    monkeypatch.setattr(resources, "_OWNS_PROCESS", False, raising=False)
    resets: list[str] = []

    class ClearRefs:
        def write_text(self, text: str) -> int:
            resets.append(text)
            return len(text)

    # The caller's peak above its current size: touch a block and free it.
    size = 256 * 1024**2
    block = mmap.mmap(-1, size)
    for offset in range(0, size, mmap.PAGESIZE):
        block[offset] = 1
    block.close()
    before = _callers_peak()
    output = tmp_path / "artifact"

    class Backend:
        def generate(self, processes, destination, *, model=None, mode="error"):
            return GenerationResult(output=output, processes=processes, mode=mode)

    monkeypatch.setattr(
        service_module, "_generator_factory", lambda config, progress: Backend()
    )
    result = Generator(GenerationConfig()).generate("d d~ > z g", output)
    with peak_memory() as region, resources.process_tree_peak_memory() as tree:
        pass

    after = _callers_peak()
    assert before[0] is not None and before[1] is not None
    assert after[0] is not None and after[1] is not None
    assert after[0] >= before[0] and after[1] >= before[1]
    monkeypatch.setattr(resources, "_PROC_CLEAR_REFS", ClearRefs())
    with peak_memory():
        pass
    assert resets == []  # never written as a library
    statistics = result.statistics
    assert statistics is not None
    assert statistics.peak_rss_bound in ("exact", "upper bound")
    assert statistics.peak_rss_method.startswith("VmHWM;")
    assert region.result is not None and region.result.method == "VmHWM"
    assert tree.result is not None and tree.result.bound in ("exact", "upper bound")


@pytest.mark.skipif(not _LINUX, reason="the resettable mark is Linux's")
def test_the_command_line_owns_its_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pyamplicol`` on the command line (``main()`` without arguments)
    owns its process; an in-process call of the command (``main(argv)``)
    does not."""

    import importlib

    from pyamplicol.reporting import resources

    cli_main = importlib.import_module("pyamplicol.cli.main")

    monkeypatch.setattr(resources, "_OWNS_PROCESS", False)
    monkeypatch.setattr(cli_main, "run_cli", lambda argv: 3)
    assert cli_main.main(["doctor"]) == 3
    assert not resources.owns_process_accounting()
    monkeypatch.setattr(cli_main, "_exit_promptly", lambda code: None)
    assert cli_main.main() == 3
    assert resources.owns_process_accounting()


def test_the_inspect_descriptions_cover_the_artifact_table(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    shutil.copytree(OLD_ARTIFACT, root)
    described = field_descriptions(inspect_artifact(root))
    for field in ("created", "generation time", "peak RAM", "integrity"):
        assert field in described
