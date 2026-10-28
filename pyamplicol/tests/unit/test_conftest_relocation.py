# SPDX-License-Identifier: 0BSD
"""The shared test harness at another location (HANDOFF §9 item 10).

``tests/conftest.py`` derives the main checkout and the loop workspace from
the environment or from the checkout's own location, never from a host
path, and a loop test whose external input or staged native runtime cannot
be found fails, naming what is missing and where it was looked for, instead
of skipping: a relocated checkout must never pass with its loop tests
skipped. The conftest is copied into a scratch checkout (with and without a
git repository) and run by pytest in a subprocess with the loop variables
removed from the environment.
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

_PROBES = """
import pytest


@pytest.mark.loop_fast
def test_loop_input(loop_sm_ufo):
    assert loop_sm_ufo.is_dir()


@pytest.mark.loop_fast
def test_loop_native(loop_native):
    assert loop_native is not None


@pytest.mark.loop_fast
def test_opt_in_input(installed_virtual_bundle):
    pass


def test_other_input(loop_gmu_ufo):
    pass
"""

_PROBE_NAMES = ("loop_input", "loop_native", "opt_in_input", "other_input")

# A stand-in for the staged native: its loader reports it missing.
_NATIVE_STUB = {
    "pyamplicol/__init__.py": "",
    "pyamplicol/api/__init__.py": "",
    "pyamplicol/api/errors.py": "class DependencyError(RuntimeError):\n    pass\n",
    "pyamplicol/loops/__init__.py": "",
    "pyamplicol/loops/native.py": (
        "from pyamplicol.api.errors import DependencyError\n\n\n"
        "def load_loop_native():\n"
        "    raise DependencyError('no _rusticol staged in this checkout')\n"
    ),
}

_PYPROJECT = """[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "loop_fast: probe",
  "loop_physics: probe",
  "loop_nightly: probe",
  "deferred_direct_r2: probe",
]
"""

# The variables the harness reads (``conftest._EXTERNAL_INPUTS``,
# ``PYAMPLICOL_LOOP_WORKSPACE``), and git's.
_REMOVED_PREFIXES = ("PYAMPLICOL_", "PYLOOP_", "GIT_")


def _checkout(root: Path, *, git: bool) -> Path:
    checkout = root / "pyamplicol"
    (checkout / "tests" / "unit").mkdir(parents=True)
    shutil.copy2(CONFTEST, checkout / "tests" / "conftest.py")
    (checkout / "tests" / "unit" / "test_probes.py").write_text(_PROBES)
    (checkout / "pyproject.toml").write_text(_PYPROJECT)
    stub = root / "stub"
    for relative, text in _NATIVE_STUB.items():
        path = stub / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    if git:
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    return checkout


def _run(checkout: Path, **environment: str) -> dict[str, tuple[str, str]]:
    """pytest on the probes; test name -> (outcome, report text)."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_REMOVED_PREFIXES)
    }
    # The harness skips loop tests without a licence before any runs; the
    # probes never touch Symbolica.
    env.update(
        SYMBOLICA_LICENSE="probe",
        PYTHONPATH=str(checkout.parent / "stub"),
        PYTHONDONTWRITEBYTECODE="1",
        **environment,
    )
    report = checkout.parent / "report.xml"
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
    outcomes = {}
    for case in ElementTree.parse(report).getroot().iter("testcase"):
        name = case.get("name", "").removeprefix("test_")
        found = [
            child for child in case if child.tag in ("error", "failure", "skipped")
        ]
        if not found:
            outcomes[name] = ("passed", "")
            continue
        (child,) = found
        outcomes[name] = (child.tag, f"{child.get('message', '')} {child.text or ''}")
    assert sorted(outcomes) == sorted(_PROBE_NAMES), result.stdout + result.stderr
    return outcomes


@pytest.mark.parametrize("git", [True, False], ids=["git", "copy"])
def test_relocated_checkout_fails_loop_tests_loudly(tmp_path, git):
    """No workspace beside the relocated checkout and no loop variable: the
    loop tests that need an input or the native fail (setup error) with the
    locations looked at; the opt-in input and the non-loop test skip with
    their reasons."""

    checkout = _checkout(tmp_path / "relocated", git=git)
    outcomes = _run(checkout)
    outcome, text = outcomes["loop_input"]
    assert outcome == "error", outcomes
    assert "loop_sm_ufo unavailable: set PYAMPLICOL_NLO_UFO_REFERENCE" in text
    assert str(checkout / "FOR_REFERENCE_ONLY") in text
    assert "no loop workspace: PYAMPLICOL_LOOP_WORKSPACE is unset" in text
    assert str(tmp_path / "relocated" / "workspace") in text
    outcome, text = outcomes["loop_native"]
    assert outcome == "error", outcomes
    assert "staged native loop runtime unavailable" in text
    assert outcomes["opt_in_input"][0] == "skipped"
    assert "set PYAMPLICOL_VIRTUAL_BUNDLE_INSTALL" in outcomes["opt_in_input"][1]
    assert outcomes["other_input"][0] == "skipped"
    assert "loop_gmu_ufo unavailable" in outcomes["other_input"][1]


def test_relocated_checkout_derives_its_workspace(tmp_path):
    """A workspace beside the relocated checkout supplies the defaults (no
    host path, no variable); ``PYAMPLICOL_LOOP_WORKSPACE`` overrides it, and a
    configured input that does not exist fails."""

    root = tmp_path / "relocated"
    checkout = _checkout(root, git=True)
    (root / "workspace/reference/sources/MG5_aMC_v3_7_0/models/loop_sm").mkdir(
        parents=True
    )
    outcomes = _run(checkout)
    assert outcomes["loop_input"][0] == "passed", outcomes
    assert outcomes["other_input"][0] == "skipped"
    assert str(root / "workspace" / "reference") in outcomes["other_input"][1]

    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "reference/models/loop_qcd_qed_sm_Gmu_py3").mkdir(parents=True)
    outcomes = _run(checkout, PYAMPLICOL_LOOP_WORKSPACE=str(elsewhere))
    assert outcomes["other_input"][0] == "passed", outcomes
    outcome, text = outcomes["loop_input"]
    assert outcome == "error", outcomes
    assert f"PYAMPLICOL_LOOP_WORKSPACE={elsewhere}" in text

    missing = tmp_path / "missing"
    outcomes = _run(checkout, PYAMPLICOL_NLO_UFO_REFERENCE=str(missing))
    outcome, text = outcomes["loop_input"]
    assert outcome == "error", outcomes
    assert f"PYAMPLICOL_NLO_UFO_REFERENCE={missing} does not exist" in text
