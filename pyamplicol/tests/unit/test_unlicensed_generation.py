# SPDX-License-Identifier: 0BSD
"""Generation without a personal Symbolica licence (DESIGN §12.1 (e)).

Every native entry point of ``pyamplicol._rusticol`` (the Python bindings of
the tree runtime and of the loop module) and of the loop SDK holds
pyAmpliCol's Symbolica library key on its calling thread, and the native
thread pools hold it on their workers; the Python side holds the package key
in the package's frames. A user without ``SYMBOLICA_LICENSE`` therefore never
takes the user's one restricted Symbolica permit: generation, check and
evaluation complete while another unlicensed Symbolica process holds it, and
Symbolica's restricted-mode banner is never printed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_fast

ROOT = Path(__file__).resolve().parents[2]
_TOOL = ROOT / "tools" / "loop_artifact_portability.py"
BANNER = "restricted Symbolica instance"
# A cheap loop process of the shipped master card whose generation reads the
# loop kernels' terms natively (``loop_kernel_terms``), the step that took
# the user's permit before every entry point held the key.
PROCESS = "u u~ > d d~ [QCD]"


def _portability():
    spec = importlib.util.spec_from_file_location("_unlicensed_portability", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(card: Path, arguments: list[str], env: dict[str, str], cwd: Path):
    return subprocess.run(
        [sys.executable, "-m", "pyamplicol", str(card), *arguments],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=1800,
    )


def test_unlicensed_loop_generation_and_evaluation_run_while_the_permit_is_held(
    tmp_path,
):
    """Generate, check and evaluate (binary64 and double-double) a loop
    process of the shipped master card without ``SYMBOLICA_LICENSE`` while
    another unlicensed Symbolica process holds the user's restricted permit
    (a private lock directory): each step ends with status 0 and no
    restricted-mode banner. The controls: a plain Symbolica process aborted
    meanwhile (``permit.record["verified"]``), and so does the same
    generation with the native key disabled (``PYAMPLICOL_NATIVE_LIBRARY_KEY
    =off``), so generation does reach the native copy of Symbolica."""

    portability = _portability()
    card = tmp_path / "loop_generate.toml"
    card.write_text((ROOT / "examples" / "loop_generate.toml").read_text())
    environment = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    environment["COLUMNS"] = "120"
    lock = tmp_path / "locks"
    lock.mkdir(mode=0o700)
    steps = (
        ["generate", f"--process={PROCESS}", "--output=./out"],
        ["check", "--output=./out", "--points=1"],
        ["evaluate", "--output=./out"],
        ["evaluate", "--output=./out", "--arithmetic=double-double"],
    )
    no_key = {**environment, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"}
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        control = _run(
            card,
            ["generate", f"--process={PROCESS}", "--output=./no-key"],
            permit.environment(no_key),
            tmp_path,
        )
        assert control.returncode != 0
        assert BANNER in control.stdout + control.stderr
        for arguments in steps:
            done = _run(card, arguments, permit.environment(environment), tmp_path)
            output = done.stdout + done.stderr
            assert done.returncode == 0, (arguments, output[-3000:])
            assert BANNER not in output, (arguments, output[-3000:])
            assert not any(m in output for m in portability.RESTRICTED_MARKERS)
    assert permit.record["verified"], permit.record
