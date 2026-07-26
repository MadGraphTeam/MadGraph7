# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_kernels import _equivalent_component_scale

ROOT = Path(__file__).resolve().parents[2]


def test_compact_component_scale_accepts_only_an_exact_sign() -> None:
    _sym._ensure_symbolica()
    compact = (_sym.E("x+1"), _sym.E("2*x-3"))

    positive = _equivalent_component_scale(compact, compact)
    negative = _equivalent_component_scale(
        tuple(-component for component in compact),
        compact,
    )
    float_positive = _equivalent_component_scale(
        (_sym.E("-1.00000000000000*(x+1)"),),
        (_sym.E("-(x+1)"),),
    )
    rescaled = _equivalent_component_scale(
        tuple(2 * component for component in compact),
        compact,
    )

    assert positive == _sym.E("1")
    assert negative == _sym.E("-1")
    assert float_positive == _sym.E("1")
    assert rescaled is None


def test_float_component_scale_fails_closed_without_symbolica_abort() -> None:
    code = """
from pyamplicol.models import compiler_symbolica as sym
from pyamplicol.models.compiler_kernels import _equivalent_component_scale

sym._ensure_symbolica()
result = _equivalent_component_scale(
    (sym.E("1.1*(x+1)"),),
    (sym.E("y+1"),),
)
assert result is None
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["SYMBOLICA_HIDE_BANNER"] = "1"
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
