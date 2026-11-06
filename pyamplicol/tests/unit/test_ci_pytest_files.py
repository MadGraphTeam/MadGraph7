# SPDX-License-Identifier: 0BSD
"""``tools/ci/run_pytest_files.py``: one pytest process per unit, counts and
the checks that keep a CI step from passing with nothing run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "ci" / "run_pytest_files.py"

_FILES = {
    "test_a.py": (
        "import pytest\n\n\n@pytest.mark.slow\ndef test_one():\n    pass\n\n\n"
        "def test_two():\n    pass\n"
    ),
    "test_b.py": "import pytest\n\n\ndef test_skip():\n    pytest.skip('probe')\n",
    "test_c.py": (
        "import pytest\n\n\n@pytest.mark.xfail(strict=True)\n"
        "def test_expected():\n    assert False\n"
    ),
    "test_d.py": "def test_fails():\n    assert False\n",
}


def _run(tmp_path: Path, *arguments: str, licensed: bool = False):
    for name, text in _FILES.items():
        (tmp_path / name).write_text(text)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: probe\n")
    env = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    env.pop("GITHUB_STEP_SUMMARY", None)
    if licensed:
        env["SYMBOLICA_LICENSE"] = "probe"
    result = subprocess.run(
        [sys.executable, str(TOOL), *arguments, "--", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout + result.stderr


def test_counts_executed_tests_including_expected_failures(tmp_path):
    code, output = _run(tmp_path, "test_a.py", "test_c.py")
    assert code == 0, output
    assert "2 units, 3 tests executed, 0 skipped, 0 failed" in output
    assert "one at a time" in output


def test_a_failing_unit_fails_the_run_with_its_log(tmp_path):
    code, output = _run(tmp_path, "test_a.py", "test_d.py")
    assert code == 1, output
    assert "::error::test_d.py exited 1" in output
    assert "::group::test_d.py (exit 1)" in output
    assert "assert False" in output


def test_an_all_skipped_unit_fails_only_with_require_each(tmp_path):
    code, output = _run(tmp_path, "test_a.py", "test_b.py")
    assert code == 0, output
    code, output = _run(tmp_path, "test_a.py", "test_b.py", "--require-each")
    assert code == 1, output
    assert "::error::test_b.py executed no test (all skipped)" in output


def test_minimum_executed_count(tmp_path):
    code, output = _run(tmp_path, "test_a.py", "--min-executed", "3")
    assert code == 1, output
    assert "::error::2 tests executed, fewer than 3" in output


def test_marker_selection_drops_files_and_only_missing_files_may_pass(tmp_path):
    code, output = _run(tmp_path, "test_*.py", "--marker", "slow")
    assert code == 0, output
    assert "-> 1 units from 4 targets" in output
    assert "1 units, 1 tests executed" in output
    for arguments in ((), ("--allow-missing", "none yet")):
        code, output = _run(tmp_path, "test_b.py", "--marker", "slow", *arguments)
        assert code == 1, output
        assert "::error::no test selected" in output
    code, output = _run(
        tmp_path, "test_new*.py", "--marker", "slow", "--allow-missing", "none yet"
    )
    assert code == 0, output
    assert "::notice::none yet" in output


@pytest.mark.parametrize("licensed", [False, True])
def test_split_shards_partition_the_test_ids(tmp_path, licensed):
    record = tmp_path / "durations.json"
    seen = []
    for shard in ("1/2", "2/2"):
        code, output = _run(
            tmp_path,
            "test_a.py",
            "test_c.py",
            "--split",
            "test_a.py",
            "--shard",
            shard,
            "--jobs",
            "2",
            "--record",
            str(record),
            licensed=licensed,
        )
        assert code == 0, output
        assert ("2 at a time" in output) is licensed
        seen += [line.split()[-1] for line in output.splitlines() if "] exit 0" in line]
        durations = json.loads(record.read_text())["durations"]
        assert set(durations) <= {"test_a.py", "test_c.py"}
        record.rename(tmp_path / f"record-{shard[0]}.json")
    assert sorted(seen) == ["test_a.py::test_one", "test_a.py::test_two", "test_c.py"]
    records = [str(tmp_path / "record-1.json"), str(tmp_path / "record-2.json")]
    code, output = _run(tmp_path, *records, "--merge", "--record", str(record))
    assert code == 0, output
    assert "2 records: 3 units, 3 tests executed, 0 skipped, 0 failed" in output
    assert set(json.loads(record.read_text())["durations"]) == {
        "test_a.py",
        "test_c.py",
    }
    code, output = _run(tmp_path, *records, "--merge", "--min-executed", "4")
    assert code == 1, output
    assert "::error::3 tests executed (at least 4 required), 0 failed" in output


def test_exclusions_apply_to_the_files_collected_from_a_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    code, output = _run(
        tmp_path, ".", "--marker", "not slow", "--exclude", "test_[bcd].py"
    )
    assert code == 0, output
    assert "-> 1 units from 1 targets" in output
    assert "1 units, 1 tests executed" in output


def test_excluded_and_unmatched_globs_select_nothing(tmp_path):
    code, output = _run(
        tmp_path, "test_[ab].py", "--exclude", "test_[ab].py", "--marker", "slow"
    )
    assert code == 1, output
    assert "::error::no test selected" in output
    assert "collected" not in output
    code, output = _run(tmp_path, "test_none*.py", "--marker", "slow")
    assert code == 1, output
    assert "collected" not in output
