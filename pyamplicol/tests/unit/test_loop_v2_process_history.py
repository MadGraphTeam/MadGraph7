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
  writer stores its evaluators), of a producer with 20,000 and with 70,000
  symbols. Symbolica orders terms by their symbols' registration indices;
  NinjaRS keeps its double-double and arbitrary-precision evaluators
  independent of them only when its positional slots and master-integral
  symbols are registered first (``ninjars_symbolica::reserve``, called by
  ``rusticol_core::symbol_order``). Without that, a generation earlier in
  the process moved later double-double NinjaRS values (here by 4e-30
  relative after 70,000 symbols, which also leave the slot pool without
  variable slots; at near-forward g g > h h points by 3e-16; after 20,000
  symbols by 1e-28) and their fallback-guard estimates (2.4e-19 -> 4.1e-16
  in test_loop_v2_fallback_guard's SDK comparison after other tests).

Each child also reports the slot regime NinjaRS obtained (two-byte variable
slots, after every history) and the registration ID of NinjaRS's first slot
symbol: the same after every history exactly when the reservation was the
first registration on every path.
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
elif history.endswith("-symbol import"):
    from symbolica import S, FunctionDefinition

    from pyamplicol.evaluators.symbolica_helpers import saved_evaluator

    for i in range(int(history.split("-")[0])):
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
from pyamplicol.loops.native import load_loop_native

native = load_loop_native()
outputs["regime"] = native.symbol_regime()
outputs["first_slot_id"] = native.symbol_first_slot_id()
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
    evaluation and after a generation's whole-state import of 20,000 and of
    70,000 symbols (module docstring), bitwise, with variable slots."""

    alone = _outputs("none")
    assert len(alone) == 8 and alone["regime"] == "variables", alone.get("regime")
    assert isinstance(alone["first_slot_id"], int), alone["first_slot_id"]
    for history in ("500 digits", "20000-symbol import", "70000-symbol import"):
        after = _outputs(history)
        moved = sorted(key for key in alone if after[key] != alone[key])
        assert not moved, (history, moved, [(alone[k], after[k]) for k in moved[:1]])
