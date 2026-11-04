# SPDX-License-Identifier: 0BSD
"""Regression tests of hard-coded limits replaced by data-derived values
(M-FINAL mf-limits-py): each case lies past a former fixed bound."""

from __future__ import annotations

from fractions import Fraction

import pytest


def test_full_colour_keeps_every_nc_power_at_any_multiplicity() -> None:
    """'Full' colour dropped N_c powers below n - 40 (a literal depth of 20)."""

    from pyamplicol.color.adjoint_kernel import _minimum_power, full_minimum_power

    assert full_minimum_power(41, None) == 0
    assert full_minimum_power(400, None) == 0
    assert full_minimum_power(41, 20) == 1  # the former floor, still selectable
    assert _minimum_power(60, "full", None) == 0
    assert _minimum_power(60, "nlc", None) == 58


def test_ct_dummy_stride_follows_the_model_indices() -> None:
    """Dummy offsets 1000/2000/3000 stay for small negative UFO dummies and
    widen when a model's dummies would collide with them."""

    from pyamplicol.models.ct_line_insertions import _dummy_stride, _inner, _offsets

    assert _dummy_stride([]) == 1000
    assert _dummy_stride(["Metric(UFO::dummy(-1),UFO::dummy(-999))"]) == 1000
    assert _offsets(1000) == (1000, 2000, 3000) and _inner(1000) == (9001, 9002)
    assert _dummy_stride(["P(UFO::dummy(-1500),1)"]) == 10000
    assert _dummy_stride(["dummy(3)", "dummy(-997)"]) == 10000
    for texts in (["dummy(-1500)"], ["dummy(3)", "dummy(-997)"], ["dummy(-60000)"]):
        stride = _dummy_stride(texts)
        values = [int(t[6:-1]) for t in texts]
        renamed = [value + offset for offset in _offsets(stride) for value in values]
        assert len(set(renamed)) == len(renamed)
        assert not set(renamed) & set(_inner(stride))


def test_fermion_pairing_catalog_has_no_default_bounds() -> None:
    """16 endpoints / 720 pairings per class / 4096 rules were default caps."""

    from pyamplicol.generation.recurrence_fermion_pairing import (
        FermionPairingLimitsV1,
    )

    limits = FermionPairingLimitsV1()
    assert (
        limits.max_endpoints,
        limits.max_pairings_per_species,
        limits.max_total_rules,
    ) == (None, None, None)
    assert FermionPairingLimitsV1(max_total_rules=5040).max_total_rules == 5040
    with pytest.raises(ValueError, match="None or a positive integer"):
        FermionPairingLimitsV1(max_endpoints=0)


def test_process_binding_takes_long_process_ids() -> None:
    """Process IDs are bounded by their u32 length field (formerly 4096 B)."""

    import pyamplicol.generation.recurrence_schedule_sharing as sharing

    with pytest.raises(sharing.RecurrenceScheduleSharingError, match="not 65536"):
        sharing._wire_u16(65536, "JIT parent component")
    assert sharing._wire_u16(65535, "JIT parent component") == 65535
    long_id = "p" * 5000
    with pytest.raises(sharing.RecurrenceScheduleSharingError) as refused:
        sharing.encode_recurrence_process_binding(
            process_id=long_id,
            schedule_digest="0" * 64,
            process_digest="0" * 64,
            process_semantic_digest="0" * 64,
            process_support_mask=1,
            remap=None,
            executor_pack=None,
        )
    # Past the length check: the next contract (the remap) is what refuses.
    assert "process ID" not in str(refused.value)


def test_simplest_rationals_have_no_denominator_cap() -> None:
    from pyamplicol._internal.physics.rationals import (
        simplest_rational_between,
        simplest_rational_for_float,
    )

    assert simplest_rational_for_float(1 / 3) == Fraction(1, 3)
    assert simplest_rational_for_float(1 / 1001) == Fraction(1, 1001)
    assert simplest_rational_for_float(-2.5e-7) == Fraction(-1, 4_000_000)
    assert simplest_rational_for_float(2.0**60) == 2**60
    assert simplest_rational_between(Fraction(31, 100), Fraction(32, 100)) == Fraction(
        5, 16
    )
    assert simplest_rational_between(Fraction(-1), Fraction(1, 3)) == 0


def test_builtin_validation_points_cover_decays() -> None:
    """Built-in validation kinematics took two beams only."""

    from pyamplicol._internal.physics.types import NativeEvaluationError
    from pyamplicol.models.builtin.validation import generic_validation_point

    points = generic_validation_point("z > u u~ g")
    assert [p.pdg for p in points] == [23, 2, -2, 21]
    assert points[0].momentum == (91.188, 0.0, 0.0, 0.0)
    for component in range(4):
        total = sum(float(p.momentum[component]) for p in points[1:])
        assert abs(total - points[0].momentum[component]) <= 1e-12 * 91.188
    with pytest.raises(NativeEvaluationError, match="rest frame"):
        generic_validation_point("z > u u~ g", sqrt_s=100.0)
    beams = generic_validation_point("u u~ > d d~ g")
    assert beams[0].momentum == (500.0, 0.0, 0.0, 500.0)
