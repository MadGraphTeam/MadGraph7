# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
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

    sample = sampler.sample(records)

    assert tuple(member.pid for member in sample.members) == (100, 101, 102, 103)
    assert sample.rss_bytes == (1024 + 2048 + 4096 + 512) * 1024


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
    assert "RSS limit exceeded" in completed.stderr
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
    assert workflow.count(guarded) == 6
    for heavy_command in (
        "dependencies/install_dependencies.py",
        "tests/unit/test_generation_execution_schema.py",
        "tests/integration/test_schema_v3_generation_runtime.py",
        "tests/integration/test_multilanguage_api.py",
        "just rust-check",
        "just rust-test",
    ):
        assert heavy_command in workflow
