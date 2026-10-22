# SPDX-License-Identifier: 0BSD
"""Schedule M equals schedule B natively on every control (O4, design §2.2, WP2.8).

Every control process (gluons, one and two quark lines, photon exchange) at
full, next-to-leading and leading colour: schedule M dresses the families
whose skeletons are certified primitive classes and schedule B the others
(generic fallback per family; ``u u~ > u u~`` has no certified family, so its
graph is empty and B runs throughout), and the native family numerators of
the two schedules agree

* bit for bit in the exact rational carrier (first point),
* within 1e-13 in binary64 and within 1e-28 in DoubleFloat (inputs formed at
  32 digits) of each family's largest sqrt(s)^r-weighted coefficient (two
  points; ``test_loop_v2_merged._assert_close``).

Points are made momentum conserving exactly (``_conserving``): M closes a
ring in its cut frame and maps it into the trie frame by a shift that
assumes conservation.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from pyamplicol.loops.v2 import merged
from pyamplicol.loops.v2.assemble import Precision
from pyamplicol.loops.v2.plan_export import NativeExecutor

from .test_loop_v2_merged import (
    PARAMETERS,
    PROCESSES,
    _assert_close,
    _assert_exact,
    _conserving,
    _points,
    slices,  # noqa: F401  (fixture)
)

CONTROLS = (
    "gg_gg",
    "uux_gg",
    "uux_ddxg",
    "uux_uux",
    "uux_uuxg",
    "uux_emepgg",
    "uux_ggg",
    "gg_ggg",
)
ACCURACIES = ("full", "nlc", "lc")
# Controls whose every family is certified, and the one with none.
EVERYWHERE = {"gg_gg", "gg_ggg"}
NOWHERE = {"uux_uux"}
FALLBACK_REASONS = {
    "generic seed (no certified primitive class)",
    "the B seed is not a certified combination",
}
CARRIERS = {
    "double": ({}, "1e-13"),
    "double-double": (
        {"precision": Precision.arbitrary(32), "double_double": True},
        "1e-28",
    ),
}


@pytest.mark.loop_physics
@pytest.mark.parametrize("accuracy", ACCURACIES)
@pytest.mark.parametrize("name", CONTROLS)
def test_m_equals_b_on_every_control(slices, name, accuracy):  # noqa: F811
    slice_ = slices(name, accuracy)
    graph = merged.merged_graph(slice_)
    families = set(slice_.plan.schedules)
    assert graph.families | set(graph.fallback) == families
    assert not graph.families & set(graph.fallback)
    assert set(graph.fallback.values()) <= FALLBACK_REASONS
    assert (not graph.fallback) == (name in EVERYWHERE)
    assert (not graph.families) == (name in NOWHERE)
    executor = NativeExecutor(slice_, forward=False, merged=graph)
    keys = executor.exported.families
    assert executor.native("double").merged_families == sorted(
        keys.index(key) for key in graph.families
    )

    points, scales = _points(PROCESSES[name][3])
    points, scales = [_conserving(p) for p in points[:2]], scales[:2]
    energies = [Fraction(p[0][0]) + Fraction(p[1][0]) for p in points]
    options = {"parameters": PARAMETERS, "exact": True}
    (b,) = executor.family_numerators(points[:1], scales[:1], **options)
    (m,) = executor.family_numerators(points[:1], scales[:1], schedule="M", **options)
    _assert_exact(m, b, (name, accuracy, "exact"))
    for carrier, (extra, tolerance) in CARRIERS.items():
        options = {"parameters": PARAMETERS, **extra}
        b = executor.family_numerators(points, scales, **options)
        m = executor.family_numerators(points, scales, schedule="M", **options)
        for index, (mine, theirs) in enumerate(zip(m, b, strict=True)):
            energy = f"{float(energies[index]):.17g}"
            label = (name, accuracy, carrier, index)
            _assert_close(mine, theirs, energy, tolerance, label)
