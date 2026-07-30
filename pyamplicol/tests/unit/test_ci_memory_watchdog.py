# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "tools/ci/memory_watchdog.py"
if WATCHDOG.is_file():
    from tools.ci import memory_watchdog as watchdog
else:  # CI-only tools are intentionally absent from unpacked source distributions.
    watchdog = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    watchdog is None,
    reason="the CI-only memory watchdog is intentionally absent from the sdist",
)


def test_argument_validation_and_default_limit() -> None:
    parser = watchdog._parser()
    parsed = parser.parse_args(("--", sys.executable, "-c", "pass"))

    assert parsed.limit_gib is None
    assert parsed.limit_mib is None
    assert watchdog.DEFAULT_LIMIT_GIB == 30.0
    with pytest.raises(SystemExit) as missing:
        watchdog.main(())
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as invalid:
        watchdog.main(("--limit-gib", "0", "--", "true"))
    assert invalid.value.code == 2


def test_linux_proc_parsers_handle_parentheses_and_rss() -> None:
    stat = "412 (worker (phase 2)) S 99 401 401 0 -1 0"
    status = "Name:\tworker\nVmPeak:\t9000 kB\nVmRSS:\t1234 kB\n"

    assert watchdog._parse_proc_stat(stat) == (99, 401)
    assert watchdog._parse_proc_status_rss(status) == 1234 * 1024
    assert watchdog._parse_proc_status_rss("Name:\tworker\n") == 0


def test_ps_parser_and_tree_sampler_include_group_and_escaped_descendants() -> None:
    records = watchdog._parse_ps_output(
        """
        100 1 100 1024
        101 100 100 2048
        102 101 102 4096
        103 1 100 512
        900 1 900 9999
        """
    )
    sampler = watchdog.ProcessTreeSampler(root_pid=100, root_pgid=100)

    footprints = {
        100: 8 * watchdog.MIB,
        101: 7 * watchdog.MIB,
        103: 6 * watchdog.MIB,
    }
    sample = sampler.sample(
        records,
        physical_footprint_probe=lambda pids: {
            pid: footprints[pid] for pid in pids if pid in footprints
        },
    )

    assert tuple(member.pid for member in sample.members) == (100, 101, 102, 103)
    assert sample.rss_bytes == (1024 + 2048 + 4096 + 512) * 1024
    assert sample.physical_footprint_bytes == (
        8 * watchdog.MIB
        + 7 * watchdog.MIB
        + 4096 * 1024  # Conservative RSS fallback for the raced-out PID.
        + 6 * watchdog.MIB
    )
    assert watchdog._guard_observation(sample) == (
        sample.physical_footprint_bytes,
        watchdog.DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON,
    )


def test_raced_out_footprint_member_falls_back_to_its_last_rss() -> None:
    records = {
        100: watchdog.ProcessInfo(100, 1, 100, 3 * watchdog.MIB),
        101: watchdog.ProcessInfo(101, 100, 100, 5 * watchdog.MIB),
    }
    sampler = watchdog.ProcessTreeSampler(root_pid=100, root_pgid=100)

    sample = sampler.sample(
        records,
        physical_footprint_probe=lambda _pids: {100: 7 * watchdog.MIB},
    )

    assert sample.rss_bytes == 8 * watchdog.MIB
    assert sample.physical_footprint_bytes == 12 * watchdog.MIB


def test_guard_observation_falls_back_to_rss_without_darwin_metric() -> None:
    sample = watchdog.MemorySample(
        rss_bytes=11 * watchdog.MIB,
        members=(),
    )
    lower_footprint = watchdog.MemorySample(
        rss_bytes=11 * watchdog.MIB,
        members=(),
        physical_footprint_bytes=9 * watchdog.MIB,
    )

    assert watchdog._guard_observation(sample) == (
        11 * watchdog.MIB,
        watchdog.RSS_LIMIT_REASON,
    )
    assert watchdog._guard_observation(lower_footprint) == (
        11 * watchdog.MIB,
        watchdog.RSS_LIMIT_REASON,
    )


def test_footprint_over_rss_terminates_with_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 43210

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            self.returncode = self.returncode or -9
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def record_signal(_pgid: int, selected_signal: int) -> None:
        process.returncode = -selected_signal

    monkeypatch.setattr(watchdog.os, "killpg", record_signal)
    monkeypatch.setattr(watchdog.os, "kill", lambda *_args: None)

    def snapshotter() -> dict[int, watchdog.ProcessInfo]:
        return {
            process.pid: watchdog.ProcessInfo(
                process.pid,
                1,
                process.pid,
                8 * watchdog.MIB,
            )
        }

    stderr = io.StringIO()

    returncode = watchdog.run_guarded(
        ("synthetic-child",),
        limit_bytes=16 * watchdog.MIB,
        poll_interval=0.001,
        terminate_grace=0,
        snapshotter=snapshotter,
        physical_footprint_probe=lambda pids: {
            pid: 20 * watchdog.MIB for pid in pids
        },
        stderr=stderr,
    )

    assert returncode == watchdog.MEMORY_LIMIT_EXIT_CODE
    assert (
        "reason="
        f"{watchdog.DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON}"
        in stderr.getvalue()
    )
    assert "rss=0.008 GiB" in stderr.getvalue()
    assert "physical_footprint=0.020 GiB" in stderr.getvalue()


def test_main_uses_rss_only_on_non_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_guarded(
        command: tuple[str, ...] | list[str],
        **kwargs: object,
    ) -> int:
        captured["command"] = tuple(command)
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(watchdog.platform, "system", lambda: "Linux")
    monkeypatch.setattr(watchdog, "run_guarded", fake_run_guarded)

    assert watchdog.main(("--limit-mib", "32", "--", "synthetic-child")) == 0
    assert captured["physical_footprint_probe"] is None
    assert captured["limit_bytes"] == 32 * watchdog.MIB


def test_darwin_libproc_parsers_extract_identity_and_rss() -> None:
    bsd = bytearray(136)
    struct.pack_into("=I", bsd, 12, 123)
    struct.pack_into("=I", bsd, 16, 45)
    struct.pack_into("=I", bsd, 100, 67)
    task = bytearray(96)
    struct.pack_into("=Q", task, 8, 987_654_321)
    rusage = bytearray(96)
    struct.pack_into("=Q", rusage, 72, 1_234_567_890)

    assert watchdog._parse_darwin_bsdinfo(bytes(bsd)) == (123, 45, 67)
    assert watchdog._parse_darwin_taskinfo_rss(bytes(task)) == 987_654_321
    assert (
        watchdog._parse_darwin_rusage_phys_footprint(bytes(rusage))
        == 1_234_567_890
    )
    with pytest.raises(ValueError, match="incomplete Darwin proc_bsdinfo"):
        watchdog._parse_darwin_bsdinfo(b"short")
    with pytest.raises(ValueError, match="incomplete Darwin proc_taskinfo"):
        watchdog._parse_darwin_taskinfo_rss(b"short")
    with pytest.raises(ValueError, match="incomplete Darwin rusage_info_v0"):
        watchdog._parse_darwin_rusage_phys_footprint(b"short")


def test_platform_probe_rejects_unsupported_hosts() -> None:
    with pytest.raises(watchdog.ProbeError, match="unsupported host"):
        watchdog.process_snapshot("Plan9")


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"},
    reason="the CI memory probe supports macOS and Linux",
)
def test_host_probe_observes_current_process() -> None:
    records = watchdog.process_snapshot()

    assert os.getpid() in records
    assert records[os.getpid()].rss_bytes > 0


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="physical-footprint probing uses Darwin libproc",
)
def test_darwin_physical_footprint_probe_observes_current_process() -> None:
    footprints = watchdog.DarwinPhysicalFootprintProbe()((os.getpid(),))

    assert footprints[os.getpid()] > 0


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_exit_code_is_propagated() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--limit-mib",
            "256",
            "--poll-interval",
            "0.02",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(23)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 23
    assert "command finished exit=23" in completed.stderr


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_low_limit_terminates_child_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "child-terminated"
    pid_path = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        """
import os
import signal
import sys
import time
from pathlib import Path

marker = Path(sys.argv[1])

def terminate(_signum, _frame):
    marker.write_text("terminated", encoding="ascii")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
payload = bytearray(96 * 1024 * 1024)
marker.with_suffix(".ready").write_text(str(len(payload)), encoding="ascii")
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        """
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
Path(sys.argv[3]).write_text(str(child.pid), encoding="ascii")
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--limit-mib",
            "64",
            "--poll-interval",
            "0.02",
            "--terminate-grace",
            "1",
            "--",
            sys.executable,
            str(parent),
            str(child),
            str(marker),
            str(pid_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == watchdog.MEMORY_LIMIT_EXIT_CODE
    assert "memory limit exceeded" in completed.stderr
    assert (
        f"reason={watchdog.RSS_LIMIT_REASON}" in completed.stderr
        or (
            "reason="
            f"{watchdog.DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON}"
            in completed.stderr
        )
    )
    assert marker.read_text(encoding="ascii") == "terminated"
    child_pid = int(pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 3
    while _pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(child_pid)


def test_ci_helper_is_not_in_wheel_or_sdist_includes() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = pyproject["tool"]["maturin"]["include"]
    paths = {entry["path"] if isinstance(entry, dict) else entry for entry in includes}

    assert not any(path == "tools/ci" or path.startswith("tools/ci/") for path in paths)


def test_tests_workflow_guards_every_heavy_validation_phase() -> None:
    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")

    assert workflow.startswith("name: Tests\n")
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in workflow
    assert "ulimit -v" not in workflow
    assert "psutil" not in workflow
    guarded = "tools/ci/memory_watchdog.py --limit-gib 30 --"
    assert workflow.count(guarded) == 8
    for heavy_command in (
        "dependencies/install_dependencies.py",
        "tests/unit/test_generation_execution_schema.py",
        "tests/integration/test_schema_v3_generation_runtime.py",
        "tests/integration/test_multilanguage_api.py",
        "just rust-check",
        "just rust-test",
    ):
        assert heavy_command in workflow
