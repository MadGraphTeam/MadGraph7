# SPDX-License-Identifier: 0BSD
"""The licence switch of the shared test harness (licensed CI jobs).

Without ``SYMBOLICA_LICENSE`` ``tests/conftest.py`` skips the loop tests;
with ``PYAMPLICOL_REQUIRE_SYMBOLICA_LICENSE=1`` they fail instead, before any
of their fixtures is set up, so a missing or expired CI secret cannot turn a
licensed job green with its loop tests skipped. The conftest is copied into
a scratch checkout and run by pytest in a subprocess; the probes never touch
Symbolica.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
SWITCH = "PYAMPLICOL_REQUIRE_SYMBOLICA_LICENSE"

_PROBES = """
from pathlib import Path

import pytest


@pytest.fixture
def touched(tmp_path_factory):
    Path(__file__).with_name("fixture-was-set-up").write_text("")


@pytest.mark.loop_fast
def test_loop_probe(touched):
    pass


@pytest.mark.loop_physics
def test_physics_probe():
    pass


def test_other_probe():
    pass
"""

_PYPROJECT = """[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "loop_fast: probe",
  "loop_physics: probe",
  "loop_nightly: probe",
  "deferred_direct_r2: probe",
]
"""


def _checkout(root: Path) -> Path:
    checkout = root / "pyamplicol"
    (checkout / "tests" / "unit").mkdir(parents=True)
    shutil.copy2(CONFTEST, checkout / "tests" / "conftest.py")
    (checkout / "tests" / "unit" / "test_probes.py").write_text(_PROBES)
    (checkout / "pyproject.toml").write_text(_PYPROJECT)
    return checkout


def _run(checkout: Path, **environment: str) -> tuple[int, dict[str, tuple], str]:
    """pytest on the probes: exit code, test name -> (outcome, text), output."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYAMPLICOL_", "SYMBOLICA_LICENSE"))
    }
    env.update(PYTHONDONTWRITEBYTECODE="1", **environment)
    report = checkout.parent / "report.xml"
    report.unlink(missing_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            f"--junitxml={report}",
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    outcomes = {}
    if report.exists():
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            name = case.get("name", "").removeprefix("test_")
            found = [
                child for child in case if child.tag in ("error", "failure", "skipped")
            ]
            if found:
                (child,) = found
                text = f"{child.get('message', '')} {child.text or ''}"
                outcomes[name] = (child.tag, text)
            else:
                outcomes[name] = ("passed", "")
    return result.returncode, outcomes, output


@pytest.mark.parametrize("switch", ["", "0", "1"])
def test_licensed_loop_tests_run(tmp_path, switch):
    checkout = _checkout(tmp_path)
    code, outcomes, output = _run(
        checkout, SYMBOLICA_LICENSE="probe", **{SWITCH: switch}
    )
    assert code == 0, output
    assert {name: outcome for name, (outcome, _) in outcomes.items()} == {
        "loop_probe": "passed",
        "physics_probe": "passed",
        "other_probe": "passed",
    }


@pytest.mark.parametrize("switch", [None, "", "0"])
def test_unlicensed_loop_tests_skip_without_the_switch(tmp_path, switch):
    checkout = _checkout(tmp_path)
    code, outcomes, output = _run(
        checkout, **({} if switch is None else {SWITCH: switch})
    )
    assert code == 0, output
    for name in ("loop_probe", "physics_probe"):
        outcome, text = outcomes[name]
        assert outcome == "skipped", outcomes
        assert "loop tests require SYMBOLICA_LICENSE" in text
    assert outcomes["other_probe"][0] == "passed"


def test_unlicensed_loop_tests_fail_with_the_switch(tmp_path):
    """Each loop test fails at setup, before its fixtures; others still run."""

    checkout = _checkout(tmp_path)
    code, outcomes, output = _run(checkout, **{SWITCH: "1"})
    assert code == 1, output
    for name in ("loop_probe", "physics_probe"):
        outcome, text = outcomes[name]
        assert outcome == "error", outcomes
        assert f"loop tests require SYMBOLICA_LICENSE ({SWITCH}=1)" in text
    assert outcomes["other_probe"][0] == "passed"
    assert not (checkout / "tests" / "unit" / "fixture-was-set-up").exists()


def test_switch_rejects_other_values(tmp_path):
    checkout = _checkout(tmp_path)
    code, _, output = _run(checkout, **{SWITCH: "yes"})
    assert code == pytest.ExitCode.USAGE_ERROR, output
    assert f"{SWITCH}='yes': expected 0 or 1" in output
