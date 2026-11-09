# SPDX-License-Identifier: 0BSD

"""NinjaRS's box mu-expansion conditioning bound through pyAmpliCol's native ``V2Reducer``.

Near-planar 2 -> 3 lanes (a g g > z z z pentagon family at lens A's planar point 44, tilted out of the plane;
the stream point itself; a Born-interfered g g > g g g pentagon at tilt 1e-8) and a regular control, each with its
exact executor geometry, binary64 tensor numerator and a reference from NinjaRS's generic engine at 340 and 420 bits
(``tests/fixtures/loop/box_mu_conditioning``, a subset of NinjaRS's ``validation/conditioning/box-mu``). Every
carrier the native exposes (the fast binary64 engine, the generic engine in double-double and at 32 and 40 digits)
must refuse a lane, typed, or agree with the reference within 1e-5 of the fixture's scale; the regular lane and the
Born-interfered one are served, and binary64 refuses the planar ones: before the bound (NinjaRS 5b6a8f57) these
lanes were served up to ``1e2`` .. ``1e17`` off in binary64 and double-double.
"""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "fixtures/loop/box_mu_conditioning/fixtures.json"
)
TOLERANCE = Decimal("1e-5")

# (arithmetic, precision in digits, engine)
CARRIERS = {
    "fast": ("double", None, "fast"),
    "double-double": ("double-double", None, "generic"),
    "arbitrary-32": ("arbitrary", 32, "generic"),
    "arbitrary-40": ("arbitrary", 40, "generic"),
}


def _fixtures():
    return json.loads(FIXTURES.read_text())["fixtures"]


def _reduce(native, fixture, carrier):
    arithmetic, precision, engine = CARRIERS[carrier]
    n = fixture["n"]

    def batch(values, columns, lanes=1):
        return native.NativeBatch.from_decimal(
            [tuple(v) for v in values],
            lanes,
            columns,
            arithmetic,
            precision,
            list(range(lanes)),
        )

    lanes = len(fixture["tensors"])
    shifts = [(c, "0") for v in fixture["shifts"] for c in v] * lanes
    masses = [tuple(m) for m in fixture["masses_squared"]] * lanes
    scales = [(fixture["mu_squared"], "0")] * lanes
    s_matrix = [(x, "0") for row in fixture["s_matrix"] for x in row] * lanes
    tensors = [tuple(z) for tensor in fixture["tensors"] for z in tensor]
    width = len(fixture["tensors"][0])
    groups = [
        (
            n,
            batch(shifts, 4 * n, lanes),
            batch(masses, n, lanes),
            batch(scales, 1, lanes),
            batch(s_matrix, n * n, lanes),
            batch(tensors, width, lanes),
        )
    ]
    reducer = native.V2Reducer(arithmetic, precision, engine)
    direct, _check = reducer.reduce(groups, poles="none")
    (result,) = direct
    statuses = dict(result.statuses)
    served = [lane for lane in range(lanes) if statuses[lane] == "success"]
    values = result.successful_values().to_decimal() if served else []
    out = {}
    for lane in range(lanes):
        out[lane] = (statuses[lane], None)
    for lane, value in zip(served, values, strict=True):
        finite = value[0] if isinstance(value[0], (list, tuple)) else value
        out[lane] = ("success", (Decimal(str(finite[0])), Decimal(str(finite[1]))))
    return out


def _scale(fixture):
    with localcontext() as context:
        context.prec = 60
        return max(
            (Decimal(re) ** 2 + Decimal(im) ** 2).sqrt()
            for re, im in fixture["references"]
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
                assert status != "success"
                continue
            re, im = (Decimal(x) for x in fixture["references"][lane])
            with localcontext() as context:
                context.prec = 60
                error = ((value[0] - re) ** 2 + (value[1] - im) ** 2).sqrt() / scale
            assert error <= TOLERANCE, (fixture["name"], lane, carrier, error)
    # The regular control and the Born-interfered lane at tilt 1e-8 are served.
    for name in ("zzz-stream-0", "gg_ggg-planar-1e-8-g0"):
        assert outcomes[(name, 0)] == "success", (name, carrier)
    if carrier == "fast":
        # Binary64 refuses the planar loop-induced lanes, typed.
        for name in ("zzz-p44-tilt-1e-9", "zzz-p44-tilt-1e-12", "zzz-stream-44"):
            assert outcomes[(name, 0)] == "Unstable", (name, outcomes[(name, 0)])
