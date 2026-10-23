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
