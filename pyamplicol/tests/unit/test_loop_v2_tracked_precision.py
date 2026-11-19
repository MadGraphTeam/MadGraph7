# SPDX-License-Identifier: 0BSD
"""The point-level tracked-precision test of NinjaRS's arbitrary-precision
lanes (NinjaRS #15, defence in depth; ``rusticol_core::loop_execution::v2::
TrackedPrecision``).

Symbolica's arbitrary-precision ``Float`` tracks the bits that survive each
operation, so every lane NinjaRS's generic engine returns carries the
absolute uncertainty ``u = |x| 2^-p_x + |y| 2^-p_y`` of its value. Per
Laurent coefficient ``k``, the point's ``U_k`` and ``A_k`` sum ``u`` and the
moduli ``|x| + |y|`` over its lanes; the point fails, typed ("NinjaRS
tracked precision: ..."), when ``max_k U_k > 2^-r max_k A_k``, with ``r``
NinjaRS's rule for its master integrals at the working precision (53 bits
at most 212 bits, a quarter above, at most half). The native executor (the
inclusive Python API and the SDKs) and the Python reference path apply the
one native rule; binary64, double-double, TIRust and the unchecked mode take
no test.

These tests use synthetic lanes (values at chosen tracked precisions, the
rule itself), the reducer's rule per carrier and the reference path's typed
failure; the HC05 ``g g > x0 g g`` points it refuses are measured in the
evidence of the change (they need a generated HC05 artifact).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.loop_fast

REASON = "NinjaRS tracked precision: the point's lanes keep "


def _lane(*coefficients):
    """A synthetic lane: per Laurent coefficient (re, im, p_re, p_im)."""
    return [(str(re), str(im), p_re, p_im) for re, im, p_re, p_im in coefficients]


def test_full_precision_lanes_pass_and_a_ruined_lane_fails(loop_native):
    rule = loop_native.reduction_tracked_refusal
    regular = _lane((1.5, -0.5, 256, 256), (0.25, 0, 256, 256), (0, 0, 256, 256))
    assert rule([regular, regular], 256) is None
    # A lane as large as the others with 4 tracked bits left (HC05 point 29
    # at 57 digits kept 3.7).
    ruined = _lane((3.0, 0, 4, 256), (0, 0, 256, 256), (0, 0, 256, 256))
    reason = rule([regular, ruined], 256)
    assert reason.startswith(REASON), reason
    assert "64 required at 256 working bits; " in reason, reason
    # The remedy for an amplitude that vanishes numerically (follow-up 4).
    assert reason.endswith(
        "an amplitude that vanishes numerically (an integrated zero) is evaluated "
        'on the default route, with the rescue stack\'s "scale" metric or with '
        "reduction_unchecked=True"
    ), reason
    # A lost imaginary part counts as well; so does a pole coefficient.
    imaginary = _lane((1.0, 0.5, 256, 30), (0, 0, 256, 256), (0, 0, 256, 256))
    assert rule([regular, imaginary], 256) is not None
    pole = _lane((0, 0, 256, 256), (1.0, 0, 20, 256), (0, 0, 256, 256))
    assert rule([regular, pole], 256) is not None


@pytest.mark.parametrize(
    ("working", "required"), [(200, 53), (256, 64), (266, 66), (333, 83)]
)
def test_the_threshold_is_ninjars_rule_at_the_working_precision(
    loop_native, working, required
):
    """``r`` is NinjaRS's rule (``required_tracked_bits``) at the working
    bits of pyAmpliCol's 40, 57, 60 and 80 digits; a point keeping about
    ``r + 1`` bits passes, about ``r - 1`` fails."""

    rule = loop_native.reduction_tracked_refusal
    regular = _lane((1, 0, working, working))
    for bits, refused in ((required + 2, False), (required - 2, True)):
        lanes = [regular, _lane((1, 0, bits, working))]
        reason = rule(lanes, working)
        assert (reason is not None) is refused, (working, bits, reason)
        if refused:
            assert f"{required} required at {working} working bits; " in reason


def test_vanishing_families_and_poles_weigh_by_their_size(loop_native):
    """Families that vanish identically are rounding noise of a few tracked
    bits at a regular point (``g g > h h``: about 1e-59 with 4 bits): summed
    over the point they weigh by their size and pass, as do identically
    vanishing poles (the largest coefficient decides); a per-lane test would
    refuse every such point at every precision."""

    rule = loop_native.reduction_tracked_refusal
    regular = _lane((2.0e-3, 1.0e-3, 256, 256), (1e-59, 0, 4, 256), (-3e-59, 0, 4, 256))
    vanishing = _lane((5e-59, 2e-59, 4, 4), (2e-59, 0, 4, 256), (0, 0, 256, 256))
    assert rule([regular, vanishing, vanishing, regular], 256) is None
    # Nothing to test: no lane, or exact zeros.
    assert rule([], 256) is None
    assert rule([_lane((0, 0, 256, 256))], 256) is None


def test_the_reducer_states_the_rule_of_its_carrier(loop_native):
    """NinjaRS's arbitrary-precision generic engine tests its lanes at its
    working bits; binary64, double-double, the unchecked mode and TIRust's
    reducer take no test. A full-precision reduction passes."""

    for arithmetic, digits, engine in (
        ("double", None, "fast"),
        ("double", None, "generic"),
        ("double-double", None, "generic"),
    ):
        assert (
            loop_native.V2Reducer(arithmetic, digits, engine).tracked_precision is None
        )
    reducer = loop_native.V2Reducer("arbitrary", 77, "generic")
    assert reducer.tracked_precision == (256, 64)
    unchecked = loop_native.V2Reducer("arbitrary", 77, "generic", unchecked=True)
    assert unchecked.tracked_precision is None

    def batch(values, columns):
        return loop_native.NativeBatch.from_decimal(
            values, 1, columns, "arbitrary", 77, [0]
        )

    # A tadpole q0 / ((q + p)^2 - m^2) at full precision.
    numerator = [("0", "0")] * 5
    numerator[1] = ("1", "0")
    group = (
        1,
        batch([("3", "0"), ("0", "0"), ("0", "0"), ("0", "0")], 4),
        batch([("29929", "0")], 1),
        batch([("1000000", "0")], 1),
        batch([("0", "0")], 1),
        batch(numerator, 5),
    )
    direct, _check = reducer.reduce([group])
    assert dict(direct[0].statuses) == {0: "success"}
    assert reducer.tracked_refusal(direct) is None
    if hasattr(loop_native, "V2TirReducer"):
        tir = loop_native.V2TirReducer()
        assert tir.tracked_precision is None
        assert tir.tracked_refusal(direct) is None


def test_the_reference_path_fails_the_point_typed():
    """The Python reference path raises the refusal as the point's
    ``ArithmeticError`` (a failed point with that reason, as a failed lane's)
    and passes a point the reducer accepts."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.assemble import require_tracked_precision

    reason = REASON + "3.7 tracked bits (...), 64 required at 256 working bits"
    seen = []

    def refusal(direct):
        seen.append(direct)
        return reason if direct == ["ruined"] else None

    reducer = SimpleNamespace(tracked_refusal=refusal)
    require_tracked_precision(reducer, ["regular"])
    with pytest.raises(ArithmeticError, match="NinjaRS tracked precision: "):
        require_tracked_precision(reducer, ["ruined"])
    assert seen == [["regular"], ["ruined"]]


@pytest.mark.loop_physics
def test_the_reference_path_applies_the_test_to_every_point(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """The Python reference path (``LoopSlice.evaluate``) hands every
    arbitrary-precision point's direct batches to its reducer's
    ``tracked_refusal``: on ``u u~ > u u~`` at 60 digits the real reducer
    states NinjaRS's rule and accepts the point; a reducer refusing it makes
    the point fail with the refusal (``ArithmeticError``)."""

    import json
    from pathlib import Path

    from pyamplicol.loops.v2.assemble import (
        LoopSlice,
        Precision,
        SliceRequest,
        _PointEvaluation,
    )
    from pyamplicol.models.loading import load_compiled_model

    data = json.loads(
        (
            Path(__file__).resolve().parents[1] / "fixtures/loop/legacy/uux_uux.json"
        ).read_text(),
        parse_float=str,
    )
    point, scale = data["points"][0], "1000000"
    request = SliceRequest(
        "u u~ > u u~",
        {"QCD": 2, "QED": 0},
        {"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model), request, loop_native
    )
    parameters = {"aS": "0.118", "MT": "173"}
    precision = Precision.arbitrary(60)
    original = _PointEvaluation.reducer
    seen = []

    class Recording:
        """The real reducer, recording (or refusing) each point's test."""

        def __init__(self, reducer, refusal):
            self.reducer, self.refusal = reducer, refusal

        def __getattr__(self, name):
            return getattr(self.reducer, name)

        def tracked_refusal(self, direct):
            seen.append((self.reducer.tracked_precision, len(direct)))
            found = self.reducer.tracked_refusal(direct)
            return found if self.refusal is None else self.refusal

    def evaluate(refusal):
        monkeypatch.setattr(
            _PointEvaluation,
            "reducer",
            lambda self: Recording(original(self), refusal),
        )
        return slice_.evaluate(point, scale, parameters=parameters, precision=precision)

    evaluate(None)
    # One test per reduction of the point, on the real reducer's rule (the
    # reference path reduces at the requested 60 digits: 200 bits, 53
    # required) over its lanes.
    assert seen and all(rule == (200, 53) and lanes > 0 for rule, lanes in seen), seen
    reason = REASON + "1.0 tracked bits (...), 53 required at 200 working bits"
    with pytest.raises(ArithmeticError, match="NinjaRS tracked precision: "):
        evaluate(reason)
