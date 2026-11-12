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


_GUARD = "let _unlock = pyamplicol::unlock();"
_BINDINGS = (
    ROOT / "rust" / "crates" / "rusticol-python" / "src",
    ROOT / "rust" / "loop-reduction" / "python" / "src",
)
_ABI_GUARDS = (
    ROOT / "rust" / "crates" / "rusticol-capi" / "src" / "lib.rs",
    ROOT / "rust" / "loop-reduction" / "sdk" / "src" / "ffi.rs",
)


def _block_end(text: str, start: int) -> int:
    """The index after the brace block opening at ``start`` (no braces occur
    in the string or character literals of these sources' signatures)."""

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    raise AssertionError("unbalanced braces")


def _body_start(text: str, at: int) -> int:
    """The index of the body's ``{`` of the function signature at ``at``
    (outside its parentheses and brackets: ``[u32; 4]``, ``(a, b)``)."""

    depth = 0
    for index in range(at, len(text)):
        char = text[index]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "{" and depth == 0:
            return index
    raise AssertionError("no function body")


def _unguarded(text: str) -> list[str]:
    """Bindings whose body does not start with the key guard: every
    ``#[pyfunction]`` and every ``#[pymethods]`` method except plain field
    accessors (``#[getter]``, ``#[setter]``)."""

    import re

    item = re.compile(
        r"((?:\s*#\[(?:[^\[\]]|\[[^\]]*\])*\])*)\s*"
        r"(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)"
    )
    missing = []

    def check(match) -> int:
        body = _body_start(text, match.end())
        if not text[body + 1 :].lstrip().startswith(_GUARD):
            missing.append(match.group(2))
        return body

    for attribute in re.finditer(r"#\[pyfunction[^\]]*\]", text):
        match = item.match(text, attribute.start())
        assert match is not None, text[attribute.start() : attribute.start() + 200]
        check(match)
    for block in re.finditer(r"#\[pymethods\]\s*impl[^{]*\{", text):
        end = _block_end(text, block.end() - 1)
        at = block.end()
        while (match := item.search(text, at, end)) is not None:
            if re.search(r"#\[(getter|setter)\b", match.group(1)):
                body = _body_start(text, match.end())
            else:
                body = check(match)
            at = _block_end(text, body)
    return missing


def test_every_native_entry_point_holds_the_library_key():
    """The Python bindings of ``_rusticol`` and of its ``_loop`` module hold
    pyAmpliCol's Symbolica library key from their first statement, and both C
    ABIs (tree runtime, loop SDK) take it in the helper every call runs
    through: no native entry point can reach Symbolica state without it."""

    missing = {
        path.name: names
        for directory in _BINDINGS
        for path in sorted(directory.glob("*.rs"))
        if (names := _unguarded(path.read_text()))
    }
    assert not missing, missing
    for path in _ABI_GUARDS:
        text = path.read_text()
        start = text.index("fn guard(")
        assert _GUARD in text[start : _block_end(text, text.index("{", start))], path
