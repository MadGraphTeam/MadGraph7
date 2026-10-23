# SPDX-License-Identifier: 0BSD
"""v2 observables (WP2.5): checks that need no slice and no native runtime.

Flavour-member leg matching against the factorial permutation search (the
test oracle) and the native batch's refusal of a missing Born row. The
physics checks are in ``test_loop_v2_observables.py``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.loop_fast


def test_member_orders_match_the_permutation_search(loop_sm_compiled_model):
    """Species-wise leg matching (``_member_orders``) yields, per species
    map, the first leg order of the factorial permutation search (the test
    oracle), in the same candidate order, for identity ports, aliases,
    conjugations and refused pairs."""

    from itertools import permutations

    from pyamplicol.loops.flavor_mappings import identity_species_candidate
    from pyamplicol.loops.v2.observables import _member_orders
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    ir = load_compiled_model(loop_sm_compiled_model).ir

    def candidates(rep, mem, orders):
        found = {}
        for order in orders:
            species = identity_species_candidate(
                tuple(rep[k].pdg for k in order), tuple(leg.pdg for leg in mem)
            )
            if species is not None:
                key = tuple(sorted(species.items()))
                if key not in found or order < found[key][1]:
                    found[key] = (len(species), order)
        return sorted(found.values())

    pairs = [
        ("d d~ > g g", ("s s~ > g g", "d~ d > g g", "d d~ > d d~")),
        ("u u~ > d d~", ("u~ u > d d~", "d d~ > u u~", "s s~ > d d~")),
        ("u d > u d", ("d u > d u", "u d > d u", "s c > s c")),
        ("u u~ > d d~ g", ("d~ d > u~ u g", "s s~ > c c~ g")),
        ("u g > u g", ("g u > g u", "c g > c g")),
    ]
    for representative, members in pairs:
        rep = build_model_process_ir(representative, ir, color_accuracy="full").legs
        initial = [k for k, leg in enumerate(rep) if leg.side == "initial"]
        final = [k for k, leg in enumerate(rep) if leg.side == "final"]
        search = [(*a, *b) for a in permutations(initial) for b in permutations(final)]
        for process in members:
            mem = build_model_process_ir(process, ir, color_accuracy="full").legs
            expected = candidates(rep, mem, search)
            assert candidates(rep, mem, _member_orders(rep, mem)) == expected, process


def test_born_rows_missing_from_a_point_are_refused():
    """A native batch takes a missing Born helicity row as zero only when
    the point's spin sources drop it (a fixed leg with another helicity)."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.plan_export import _born_row

    row, other = ((0, 1), (1, -1)), ((0, -1), (1, -1))
    evaluation = SimpleNamespace(amplitudes={row: {"born": 1}}, fixed={1: 1})
    assert _born_row(evaluation, row) == {"born": 1}
    assert _born_row(evaluation, other) == {}
    with pytest.raises(KeyError, match="missing from the point evaluation"):
        _born_row(SimpleNamespace(amplitudes={row: {}}, fixed={}), other)


# A fresh interpreter creates ``argv[1]`` unrelated Symbolica symbols, then a
# synthetic tree stage of 400 positional inputs, without (positional path of
# ``_canonical_stage``) and with a retained model function (general path),
# and prints, per path, the raw and canonical outputs' normal forms and the
# canonical evaluator's binary64 bits at fixed inputs.
_STAGE_PROBE = """
import json
import sys
from types import SimpleNamespace

from symbolica import Expression, S

for k in range(int(sys.argv[1])):
    S(f"history::unrelated{k}")

from pyamplicol.loops.v2.assemble import _canonical_stage, _stage_evaluator

p = [S(f"probe::p{k}") for k in range(400)]
f, x, y = S("probe::f"), S("probe::x"), S("probe::y")
zero = Expression.num(0)
plain = [
    sum((p[k] * p[(k + j) % 400] * (k + 1) / 7 for k in range(400)), zero)
    for j in (1, 37)
]
called = [
    sum((f(p[k], p[(k + j) % 400]) * (k + 2) / 3 for k in range(400)), zero)
    for j in (1, 37)
]
inputs = [complex((k % 13) - 6.1, (k % 7) / 3.3) for k in range(400)]
out = {}
for name, outputs, functions in (
    ("positional", plain, ()),
    ("general", called, ((f, (x, y), x * y + x**2 / 5),)),
):
    stage = SimpleNamespace(
        parameter_symbols=p, symbolica_functions=functions, output_expressions=outputs
    )
    canonical, _parameters, _functions = _canonical_stage(stage)
    values = _stage_evaluator(stage).evaluate_complex([inputs])[0]
    out[name] = {
        "raw": [str(e) for e in outputs],
        "canonical": [str(e) for e in canonical],
        "bits": [[complex(z).real.hex(), complex(z).imag.hex()] for z in values],
    }
print(json.dumps(out))
"""


def test_canonical_stages_do_not_depend_on_the_process_history():
    """Both renaming paths of ``assemble._canonical_stage`` (positional
    inputs, and a stage with a retained model function) give the same normal
    forms and binary64 evaluator bits after 0 and 300 unrelated symbols,
    while the raw stages' normal forms differ (the probe discriminates).
    ``test_generation_does_not_depend_on_the_process_history`` checks a
    generated process."""

    import json
    import subprocess
    import sys
    from pathlib import Path

    runs = [
        subprocess.Popen(
            [sys.executable, "-c", _STAGE_PROBE, str(unrelated)],
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for unrelated in (0, 300)
    ]
    outputs = []
    for run in runs:
        out, err = run.communicate(timeout=300)
        assert run.returncode == 0, err[-4000:]
        outputs.append(json.loads(out.strip().splitlines()[-1]))
    for path in ("positional", "general"):
        first, second = (output[path] for output in outputs)
        assert first["raw"] != second["raw"], path
        assert first["canonical"] == second["canonical"], path
        assert first["bits"] == second["bits"], path


def test_misaligned_canonical_blocks_are_refused():
    """A block of canonical symbols that does not start at a multiple of
    2^16 ids would order its symbols by their absolute ids:
    ``_check_canonical_block`` refuses it, for any offset."""

    from symbolica import Expression, S

    from pyamplicol.loops.v2.assemble import _canonical_symbols, _check_canonical_block

    _canonical_symbols(1)
    for shift in (1, 256, 4097):
        for k in range(shift):
            S(f"pyamplicol_test_misaligned::shift{shift}_{k}")
        offset = len(Expression.get_all_symbol_names()) % (1 << 16)
        block = [S(f"pyamplicol_test_misaligned::b{shift}_{k}") for k in range(1 << 16)]
        assert offset
        with pytest.raises(RuntimeError, match="misaligned"):
            _check_canonical_block(block)
