# SPDX-License-Identifier: 0BSD
"""The stored form of Symbolica saved evaluators.

Every generation writer stores a Symbolica evaluator through
``symbolica_helpers.saved_evaluator``: the native's canonical form, whose bytes
depend on the evaluator alone (not on the installed Python Symbolica nor on
the symbols the producing process created before).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyamplicol.evaluators import symbolica_helpers as helpers
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppParameterKind,
    NativeDirectCppSpec,
    render_native_direct_cpp,
)
from pyamplicol.evaluators.native_eager_direct_cpp import (
    render_native_eager_direct_table_cpp,
)
from pyamplicol.evaluators.symbolica_adapters import _CompiledComplexEvaluatorAdapter
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from pyamplicol.loops.native import load_loop_native

# One evaluator calling pyAmpliCol's scalar predicates (registered function
# heads, each process creates them as an aligned block of symbol ids) and
# built-ins, saved in a process with or without an unrelated history first:
# more than a block of symbols in other namespaces, so the predicates take
# other ids.
_PRODUCER = r"""
import sys
from symbolica import Expression, S
from pyamplicol.evaluators.symbolica_helpers import saved_evaluator
from pyamplicol.loops.scalar_frontend import scalar_predicates

if sys.argv[1] == "history":
    for k in range(3):
        S(*(f"pyamplicol_saved_test_history{k}::s{i}" for i in range(25000)))
predicates = scalar_predicates()
x, y = S("pyamplicol_saved_test::x", "pyamplicol_saved_test::y")
evaluator = Expression.evaluator_multiple(
    [predicates["less"](x, y) * x.sqrt() + predicates["equal"](x, y) + Expression.PI * y.log()],
    [x, y],
    jit_compile=False,
    n_cores=1,
)
print(bytes(evaluator.save()).hex(), saved_evaluator(evaluator).hex())
"""


def _produce(history: str) -> tuple[bytes, bytes]:
    result = subprocess.run(
        [sys.executable, "-c", _PRODUCER, history],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "SYMBOLICA_HIDE_BANNER": "1"},
        timeout=300,
    )
    raw, stored = result.stdout.split()
    return bytes.fromhex(raw), bytes.fromhex(stored)


def test_stored_saves_do_not_depend_on_the_process_history() -> None:
    """The installed Symbolica's raw saves differ with the history (the
    registry numbers the referenced symbols; PyPI 3.0.3 also writes the whole
    state); the stored bytes do not, and they load to the same values."""

    from symbolica import Evaluator

    from pyamplicol.loops.scalar_frontend import scalar_predicates

    plain_raw, plain = _produce("plain")
    history_raw, history = _produce("history")
    assert plain_raw != history_raw
    assert plain == history
    assert len(plain) <= min(len(plain_raw), len(history_raw))
    native = load_loop_native()
    assert native.canonical_saved_evaluator(plain) == plain
    for raw in (plain_raw, history_raw):
        assert native.canonical_saved_evaluator(raw) == plain
    scalar_predicates()
    rows = np.array([[0.75, 1.5], [2.0, 2.0]], dtype=np.complex128)
    assert np.array_equal(
        np.asarray(Evaluator.load(plain).evaluate_complex(rows)),
        np.asarray(Evaluator.load(plain_raw).evaluate_complex(rows)),
    )


def test_saves_without_symbolica_state_are_stored_as_they_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Saved:
        @staticmethod
        def save() -> bytes:
            return b"raw bincode"

    def no_native() -> Any:
        raise AssertionError("a save without the envelope needs no native")

    monkeypatch.setattr("pyamplicol.loops.native.load_loop_native", no_native)
    assert helpers.saved_evaluator(Saved()) == b"raw bincode"


def _evaluator() -> Any:
    from symbolica import E, S

    x = S("pyamplicol_saved_writer::x")
    return E("1i/sqrt(2)*pyamplicol_saved_writer::x").evaluator(
        [x], iterations=0, n_cores=1, jit_compile=False
    )


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """The native's canonical forms handed out while the test runs."""

    native = load_loop_native()
    handed: list[bytes] = []

    class Native:
        @staticmethod
        def canonical_saved_evaluator(saved: bytes) -> bytes:
            handed.append(b"canonical:" + bytes(native.canonical_saved_evaluator(saved)))
            return handed[-1]

    monkeypatch.setattr("pyamplicol.loops.native.load_loop_native", lambda: Native)
    return handed


def test_compiled_complex_writer_stores_the_canonical_form(
    tmp_path: Path, stored: list[bytes]
) -> None:
    adapter = _CompiledComplexEvaluatorAdapter(
        _evaluator(),
        SymbolicaEvaluatorSettings(
            backend="compiled-complex", compiled_output_dir=str(tmp_path)
        ),
        "saved writer",
        input_len=1,
        output_len=1,
        compile_dense=False,
    )
    assert adapter.evaluator_state_path is not None
    assert stored and adapter.evaluator_state_path.read_bytes() == stored[-1]


def test_direct_cpp_writers_digest_the_canonical_form(stored: list[bytes]) -> None:
    direct = render_native_direct_cpp(
        _evaluator(),
        NativeDirectCppSpec(
            function_name="saved_writer",
            parameter_kinds=(NativeDirectCppParameterKind.COMPLEX_PLANE,),
            output_count=1,
            target_triple="aarch64-apple-darwin",
            cpu_features=("neon",),
            simd_lane_width=2,
        ),
    )
    assert len(stored) == 1
    assert direct.evaluator_state_sha256 == hashlib.sha256(stored[0]).hexdigest()
    eager = render_native_eager_direct_table_cpp(
        _evaluator(),
        kernel_id=7,
        input_complex_count=1,
        output_complex_count=1,
        target_triple="unit-test-target",
    )
    assert len(stored) == 2
    assert eager.evaluator_state_sha256 == hashlib.sha256(stored[1]).hexdigest()
