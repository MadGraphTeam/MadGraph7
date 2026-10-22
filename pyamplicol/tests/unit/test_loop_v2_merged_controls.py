# SPDX-License-Identifier: 0BSD
"""Schedule M equals schedule B natively on every control (O4, design §2.2, WP2.8).

Every control process (gluons, one and two quark lines, photon exchange) at
full, next-to-leading and leading colour: schedule M dresses the families
whose skeletons are certified primitive classes and schedule B the others
(generic fallback per family; ``u u~ > u u~`` has no certified family and
runs B throughout), and the native family numerators of the two schedules
agree

* bit for bit in the exact rational carrier,
* within 1e-13 in binary64 and within 1e-28 in DoubleFloat (inputs formed at
  32 digits) of each family's largest sqrt(s)^r-weighted coefficient
  (families that vanish identically by Furry's theorem against the point's
  largest family, as ``test_loop_v2_merged.test_binary64_m_equals_b``
  explains).

Points are made momentum conserving exactly (``_conserving``).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from pyamplicol.loops.v2 import merged
from pyamplicol.loops.v2.assemble import Precision
from pyamplicol.loops.v2.plan_export import NativeExecutor, _degree

from .test_loop_v2_merged import (
    PARAMETERS,
    PROCESSES,
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
# Controls whose every family is certified, and those with none.
EVERYWHERE = {"gg_gg", "gg_ggg"}
NOWHERE = {"uux_uux"}
FALLBACK_REASONS = {
    "generic seed (no certified primitive class)",
    "the B seed is not a certified combination",
}
TOLERANCES = {"double": 1e-13, "double-double": 1e-28}


def _magnitude(value) -> float:
    if isinstance(value, complex):
        return abs(value)
    return abs(complex(float(value.real), float(value.imag)))


def _assert_close(left, right, energy: float, tolerance: float, label) -> None:
    """``left`` within ``tolerance`` of ``right`` on each family's largest
    sqrt(s)^r-weighted coefficient; families below ``tolerance`` of the
    point's largest family (identically vanishing) against that one."""

    def weighted(numerator):
        return max(
            _magnitude(x) * energy ** _degree(k)
            for column in numerator.columns
            for k, x in enumerate(column)
        )

    assert left.keys() == right.keys(), label
    largest = max(weighted(numerator) for numerator in right.values())
    for key, mine in left.items():
        theirs = right[key]
        scale = weighted(theirs)
        if scale < tolerance * largest:
            scale = largest
        for x, y in zip(mine.columns, theirs.columns, strict=True):
            for k, (a, c) in enumerate(zip(x, y, strict=True)):
                error = _magnitude(a - c) * energy ** _degree(k)
                assert error <= tolerance * scale, (label, key, k, error / scale)


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
    energies = [float(Fraction(p[0][0]) + Fraction(p[1][0])) for p in points]
    runs = {
        "exact": {"parameters": PARAMETERS, "exact": True},
        "double": {"parameters": PARAMETERS},
        "double-double": {
            "parameters": PARAMETERS,
            "precision": Precision.arbitrary(32),
            "double_double": True,
        },
    }
    for carrier, options in runs.items():
        b = executor.family_numerators(points, scales, **options)
        m = executor.family_numerators(points, scales, schedule="M", **options)
        for index, (mine, theirs) in enumerate(zip(m, b, strict=True)):
            label = (name, accuracy, carrier, index)
            if carrier == "exact":
                _assert_exact(mine, theirs, label)
            else:
                _assert_close(
                    mine, theirs, energies[index], TOLERANCES[carrier], label
                )
