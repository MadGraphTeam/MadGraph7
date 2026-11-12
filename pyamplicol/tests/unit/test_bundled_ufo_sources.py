"""Every bundled UFO model's Python compiles without compile-time warnings.

Python >= 3.12 reports an invalid escape sequence in a non-raw string
literal (``'\\lambda'``) as a ``SyntaxWarning`` when the module is compiled
on first import (3.11: a ``DeprecationWarning``, hidden by default), which a
user then sees on the terminal. The bundled models must compile cleanly on
every supported interpreter, so both categories are errors here.
"""

from __future__ import annotations

import warnings
from importlib import resources
from pathlib import Path

import pytest

UFO_ROOT = Path(str(resources.files("pyamplicol").joinpath("assets", "models", "ufo")))
SOURCES = sorted(UFO_ROOT.rglob("*.py"))


def test_bundled_ufo_sources_are_found() -> None:
    assert len({path.parent for path in SOURCES}) >= 2
    assert (UFO_ROOT / "loop_qcd_qed_sm" / "parameters.py") in SOURCES


@pytest.mark.parametrize(
    "path", SOURCES, ids=[path.relative_to(UFO_ROOT).as_posix() for path in SOURCES]
)
def test_bundled_ufo_source_compiles_without_warnings(path: Path) -> None:
    source = path.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        warnings.simplefilter("error", DeprecationWarning)
        compile(source, str(path), "exec", dont_inherit=True)
