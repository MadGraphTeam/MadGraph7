# SPDX-License-Identifier: 0BSD

"""NinjaRS's measured decision of flagged lanes through pyAmpliCol's native ``V2Reducer``.

Near-forward 2 -> 2 box families (g g > z z at theta 1e-4 rotated and boosted, g g > w+ w- at 1 TeV with pT(W)
1.6 GeV, g g > h h at the fallback guard's point 2 with its production S-matrix) and a regular control, each with its
exact executor geometry, binary64 (or 60-digit) tensor numerators and a reference from NinjaRS's generic engine at
340 and 420 bits on the same inputs (``tests/fixtures/loop/lane_conditioning``, a subset of NinjaRS's
``validation/conditioning/lane``). NinjaRS's lane conditioning screen flags the near-forward families; every
NinjaRS engine pyAmpliCol builds evaluates a flagged family again at a higher internal precision on consistent
kinematics and refuses a lane, typed, only when its value deviates from that measurement by more than 2^-17 of the
family's scale (NinjaRS ``docs/lane-conditioning.md``; DECISIONS 2026-11-10 10:18). Every carrier the native exposes
(the fast binary64 engine, also unchecked, the generic engine in double-double and at 32 digits) must refuse a
lane, typed, or agree with the reference within 1e-5 of the fixture's scale; the control is served; binary64
refuses lanes of the z z and w+ w- families (served up to 1e10 off before), and double-double refuses lanes of the
h h family, whose production S-matrix the binary64 inputs' shell defect moves (served 9e11 off before).
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures/loop/lane_conditioning/fixtures.json"
)
TOLERANCE = Decimal("1e-5")

# (arithmetic, precision in digits, engine, unchecked)
CARRIERS = {
    "fast": ("double", None, "fast", False),
    "fast-unchecked": ("double", None, "fast", True),
    "double-double": ("double-double", None, "generic", False),
    "arbitrary-32": ("arbitrary", 32, "generic", False),
}
REFUSALS = {"Unstable", "IllConditioned"}


def _fixtures():
    return json.loads(FIXTURES.read_text())["fixtures"]


def _reduce(native, fixture, carrier):
    arithmetic, precision, engine, unchecked = CARRIERS[carrier]
    n = fixture["n"]
    lanes = len(fixture["tensors"])

    def batch(values, columns):
        return native.NativeBatch.from_decimal(
            [tuple(v) for v in values],
            lanes,
            columns,
            arithmetic,
            precision,
            list(range(lanes)),
        )

    shifts = [(c, "0") for v in fixture["shifts"] for c in v] * lanes
    masses = [tuple(m) for m in fixture["masses_squared"]] * lanes
    scales = [(fixture["mu_squared"], "0")] * lanes
    s_matrix = [(x, "0") for row in fixture["s_matrix"] for x in row] * lanes
    tensors = [tuple(z) for tensor in fixture["tensors"] for z in tensor]
    width = len(fixture["tensors"][0])
    groups = [
        (
            n,
            batch(shifts, 4 * n),
            batch(masses, n),
            batch(scales, 1),
            batch(s_matrix, n * n),
            batch(tensors, width),
        )
    ]
    reducer = native.V2Reducer(arithmetic, precision, engine, unchecked=unchecked)
    direct, _check = reducer.reduce(groups, poles="none")
    (result,) = direct
    statuses = dict(result.statuses)
    served = [lane for lane in range(lanes) if statuses[lane] == "success"]
    values = result.successful_values().to_decimal() if served else []
    out = {lane: (statuses[lane], None) for lane in range(lanes)}
    for lane, value in zip(served, values, strict=True):
        finite = value[0] if isinstance(value[0], (list, tuple)) else value
        out[lane] = ("success", (Decimal(str(finite[0])), Decimal(str(finite[1]))))
    return out


def _scale(fixture):
    with localcontext() as context:
        context.prec = 60
        return max(
            (Decimal(re) ** 2 + Decimal(im) ** 2).sqrt() for re, im in fixture["references"]
        )


@pytest.mark.parametrize("carrier", sorted(CARRIERS))
def test_every_lane_is_refused_or_within_the_target(loop_native, carrier):
    outcomes = {}
    for fixture in _fixtures():
        result = _reduce(loop_native, fixture, carrier)
        scale = _scale(fixture)
        for lane, (status, value) in result.items():
            outcomes[(fixture["name"], lane)] = status
            if value is None:
                assert status in REFUSALS, (fixture["name"], lane, carrier, status)
                continue
            re, im = (Decimal(x) for x in fixture["references"][lane])
            with localcontext() as context:
                context.prec = 60
                error = ((value[0] - re) ** 2 + (value[1] - im) ** 2).sqrt() / scale
            assert error <= TOLERANCE, (fixture["name"], lane, carrier, error)

    def refused(name):
        return [lane for (fixture, lane), s in outcomes.items() if fixture == name and s != "success"]

    # The regular control is served.
    assert not refused("zz-cm-theta-1"), carrier
    if carrier.startswith("fast"):
        # Binary64 refuses lanes of the near-forward z z and w+ w- families, typed,
        # also unchecked (the measured decision does not read the noise thresholds).
        for name in ("zz-boosted-theta-1e-4", "ww-1tev-pt-1.6"):
            assert refused(name), (name, carrier)
    else:
        # Above binary64 the measurement on consistent kinematics refuses the
        # h h lanes that the production S-matrix moves (lit-fgh B point 2).
        assert refused("hh-forward-2-60digits-B"), carrier
