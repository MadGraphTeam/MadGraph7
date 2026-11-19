# SPDX-License-Identifier: 0BSD
"""An evaluation's outputs do not depend on what the process did before.

The shipped loop-induced self-test artifact (g g > h) is evaluated in every
carrier (binary64, double-double, 57 digits) on both backends with the
recommended rotation check, in a fresh child process, alone and after a
history, and every output (values, statuses, stability estimates, reduction
records: the result's ``repr``) must be the same:

* a 500-digit evaluation of the same artifact first (the default arbitrary
  precision, which the per-lane rescue and the default rescue ladder mix
  into ordinary calls);
* a generation's import of a producer's whole Symbolica state first
  (``evaluators.symbolica_helpers.saved_evaluator``, as every generation
  writer stores its evaluators: a producer with 70,000 symbols). NinjaRS
  builds its double-double and arbitrary-precision evaluators on positional
  slots, which a process gets only while it has fewer than about 61,000
  symbols when the slots are first needed; afterwards it gets function slots,
  whose evaluators round differently (``rusticol_core::symbol_order``). With
  the slots built only at the first NinjaRS reduction, a generation earlier
  in the process moved later double-double NinjaRS values (here by 4e-30
  relative, at near-forward g g > h h points by 3e-16) and their
  fallback-guard estimates (2.4e-19 -> 4.1e-16 in
  test_loop_v2_fallback_guard's SDK comparison after other tests).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_CHILD = r"""
import json
import sys
from decimal import Decimal

sys.path[:] = json.loads(sys.argv[1])
history = sys.argv[2]

from pyamplicol import Runtime
from pyamplicol._internal import selftest_virtual_li as selftest
from pyamplicol.api.virtual_results import VirtualStabilityChecks

root = selftest.fixture_root()
expected = json.loads((root / "expected.json").read_text())
runtime = Runtime.load(
    root / "artifact", process=selftest.NAME, model_parameters=expected["parameters"]
)
points = [[[str(Decimal(c)) for c in p] for p in expected["points"][0]]]
scales = [expected["mu_r2"][0]]


def evaluate(arithmetic, precision, reduction, checks=True):
    return runtime.evaluate(
        points,
        mu_r2=scales,
        arithmetic=arithmetic,
        precision=precision,
        reduction=reduction,
        poles="none",
        on_failure="return",
        stability_checks=VirtualStabilityChecks(rotations=1) if checks else None,
    )


if history == "500 digits":
    evaluate("arbitrary", 500, "ninjars", checks=False)
elif history == "whole-state import":
    from symbolica import S, FunctionDefinition

    from pyamplicol.evaluators.symbolica_helpers import saved_evaluator

    for i in range(70000):
        S(f"process_history::x{i}")
    x, y, f, a = (S(f"process_history::{n}") for n in ("p", "q", "f", "a"))
    function = FunctionDefinition(f, [a], a * a + 1, inlining="never")
    body = (f(x) + y).evaluator([x, y], functions=[function], jit_compile=False)
    saved_evaluator(body)
outputs = {}
carriers = (("double", None), ("double-double", None), ("arbitrary", 57))
for arithmetic, precision in carriers:
    for reduction in ("ninjars", "tirust"):
        outputs[f"{arithmetic} {precision} {reduction}"] = repr(
            evaluate(arithmetic, precision, reduction)
        )
print(json.dumps(outputs))
"""


def _outputs(history: str) -> dict[str, str]:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(sys.path), history],
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
        cwd=ROOT,
        env=dict(os.environ),
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.loop_fast
def test_outputs_do_not_depend_on_the_process_history(loop_native):
    """Every output alone equals the same output after a 500-digit
    evaluation and after a generation's whole-state import (module
    docstring), bitwise."""

    alone = _outputs("none")
    assert len(alone) == 6
    for history in ("500 digits", "whole-state import"):
        after = _outputs(history)
        moved = sorted(key for key in alone if after[key] != alone[key])
        assert not moved, (history, moved, [(alone[k], after[k]) for k in moved[:1]])
