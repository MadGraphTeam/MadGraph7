# SPDX-License-Identifier: 0BSD
"""Stored phase-space points sample the whole phase space (USER 2026-11-14 ~20:50
UTC): no stored point set that tests, benchmarks, profiles or stability tools
read holds a degenerate point (planar, collinear or soft at the precision it is
stored at, ``tools/point_degeneracy.py``) unless its file declares it an
intended limit point with the limit and the expected behaviour
(``intended_limits``). New points come from the seeded flat generator."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import point_degeneracy as pd  # noqa: E402


def test_stored_point_sets_hold_no_unmarked_degenerate_point():
    found = pd.violations(ROOT)
    assert not found, "\n".join(f"{f} {w}: {', '.join(k)}" for f, w, k in found)


def test_the_classifier_finds_each_degeneracy_at_the_stored_precision():
    half = "500"
    beams = [[half, "0", "0", half], [half, "0", "0", "-" + half]]
    # A planar 2 -> 3 point: every momentum in the x-z plane (exact).
    planar = [
        *beams,
        ["400", "200", "0", "280"],
        ["325", "-125", "0", "-260"],
        ["275", "-75", "0", "-20"],
    ]
    assert pd.classify(planar) == ["planar"]
    # Out of the plane: generic.
    generic = [
        *beams,
        ["400", "200", "70", "280"],
        ["325", "-125", "-70", "-260"],
        ["275", "-75", "0", "-20"],
    ]
    assert pd.classify(generic) == []
    # The same planar point rounded to binary64 is planar to its rounding,
    # with a tilt well above the rounding it is not.
    rounded = [[float(Fraction(x)) for x in leg] for leg in planar]
    assert pd.classify(rounded) == ["planar"]
    tilted = [list(leg) for leg in rounded]
    tilted[2][2] += 1e-6
    tilted[4][2] -= 1e-6
    assert "planar" not in pd.classify(tilted)
    # A final particle along the beam (forward) and a soft one.
    forward = [*beams, ["300", "0", "0", "300"], ["700", "0", "0", "-300"]]
    assert "collinear 0 2" in pd.classify(forward)
    soft = [*beams, ["500", "300", "0", "400"], ["500", "-300", "0", "-400"]]
    soft.append([0.0, 0.0, 0.0, 0.0])
    assert "soft 4" in pd.classify(soft)
    # The back-to-back final pair of a 2 -> 2 point and the beams are no
    # degeneracy; a 2 -> 2 point has no planarity (four legs).
    assert (
        pd.classify([*beams, ["500", "300", "0", "400"], ["500", "-300", "0", "-400"]])
        == []
    )


def test_flat_points_are_never_degenerate():
    from pyamplicol.generation.phase_space import flat_final_states

    beams = [[500.0, 0.0, 0.0, 500.0], [500.0, 0.0, 0.0, -500.0]]
    for masses in ([0.0] * 3, [125.0, 0.0, 0.0], [91.1876] * 3, [0.0] * 4):
        for state in flat_final_states(masses, sqrt_s=1000.0, seeds=range(200)):
            assert pd.classify([*beams, *map(list, state.momenta)]) == []


def test_intended_limits_need_a_reason():
    data = {"intended_limits": {"/points[0]": "forward-angle scan"}, "points": []}
    assert pd.intended_limits(data) == {"/points[0]": "forward-angle scan"}
    assert pd.intended_limits({"points": []}) == {}
