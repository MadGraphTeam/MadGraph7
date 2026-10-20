# SPDX-License-Identifier: 0BSD
"""Independent transport controls; synthetic rows are not a Bose physics oracle.

Distinct off-shell labels and a three-cycle expose inverse, missing and doubled
leg transport. Actual local-rule admission and physical reconstruction are
separate gates; these tests deliberately construct only the admitted boundary.
"""

from copy import copy, deepcopy
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from pyamplicol import CorrelatedRequest, Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.generation.process_grouping import LegMapping
from pyamplicol.loops.flavor_transport import (
    IdentityFlavorTransport,
    NativeIdentityFlavorProgram,
)
from pyamplicol.loops.physical_source import NativeSourceModelProgram

from . import test_loop_bose_field_transport as field_tests
from .test_public_virtual_directed import _directed


@pytest.fixture
def field_control():
    return field_tests.field_control.__wrapped__()


def _recording_source(permutation=(0, 1, 3, 4, 2)):
    """A real source-wrapper type with recording numerical endpoints only."""
    prepared = object()
    request = SimpleNamespace(external_pdgs=(1, -1, 21, 21, 21))
    calls = []
    result = object()

    def evaluate(momenta, scales):
        calls.append((momenta, scales))
        return result

    virtual = SimpleNamespace(
        prepared=prepared,
        evaluate_ordered_native=evaluate,
        evaluate_ordered_decimal=evaluate,
    )
    source = NativeSourceModelProgram(
        SimpleNamespace(state=object()), virtual, matched=()
    )
    relation = IdentityFlavorTransport(
        symmetry=None,  # No symbolic proof is claimed by this transport fixture.
        mapping=LegMapping(permutation, False, "transport-only-control"),
        representative_request=request,
        member_request=request,
        representative_plan=prepared,
        member_plan=object(),
        orders=object(),
    )
    return relation, source, calls, result


def _labelled_points():
    points = np.asarray(
        [
            [
                [
                    f"{row}{leg}{component}.00000000000000000000000000000001"
                    for component in range(4)
                ]
                for leg in range(5)
            ]
            for row in range(3)
        ],
        dtype=object,
    )
    points[1, 4, 2] = Decimal("4.0000000000000000000000000000000000000007")
    return points


@pytest.mark.parametrize(
    "method", ["evaluate_ordered_native", "evaluate_ordered_decimal"]
)
def test_source_three_cycle_applies_once_without_rounding(method):
    relation, source, calls, result = _recording_source()
    mapped = relation.bind(source)
    points = _labelled_points()
    original = points.copy()
    scales = ["1e-400", "900000.00000000000000001", "1000000"]
    assert getattr(mapped, method)(points, scales) is result
    assert len(calls) == 1
    actual, actual_scales = calls[0]
    # Explicit expected ports, independent of the production permutation loop.
    np.testing.assert_array_equal(actual[:, 0], points[:, 0])
    np.testing.assert_array_equal(actual[:, 1], points[:, 1])
    np.testing.assert_array_equal(actual[:, 2], points[:, 3])
    np.testing.assert_array_equal(actual[:, 3], points[:, 4])
    np.testing.assert_array_equal(actual[:, 4], points[:, 2])
    assert actual[1, 3, 2] is points[1, 4, 2]
    assert actual_scales is scales
    np.testing.assert_array_equal(points, original)
    assert not np.array_equal(actual, points)
    assert not np.array_equal(actual, points[:, (0, 1, 4, 2, 3), :])


def test_source_cycle_commutes_with_subset_reordering_and_keeps_shaped_empty():
    relation, source, calls, _ = _recording_source()
    mapped = relation.bind(source)
    points = _labelled_points()
    mapped.evaluate_ordered_decimal(points, ["11", "22", "33"])
    all_points = calls[-1][0]
    selected = [2, 0]
    scales = ["33", "11"]
    mapped.evaluate_ordered_decimal(points[selected], scales)
    np.testing.assert_array_equal(calls[-1][0], all_points[selected])
    assert calls[-1][1] is scales
    mapped.evaluate_ordered_decimal(points[:0], [])
    assert calls[-1][0].shape == (0, 5, 4)
    with pytest.raises(ValueError, match="batch, leg"):
        mapped.evaluate_ordered_decimal(points[:, :4], scales)
    assert len(calls) == 3


def test_identity_source_passes_original_inputs_without_new_copy():
    relation, source, calls, result = _recording_source(tuple(range(5)))
    points, scales = _labelled_points(), ["11", "22", "33"]
    assert relation.bind(source).evaluate_ordered_decimal(points, scales) is result
    assert calls[0][0] is points and calls[0][1] is scales


def test_direct_constructor_cannot_bypass_nonidentity_directed_gate():
    relation, source, _, _ = _recording_source()
    bound = []
    directed = SimpleNamespace(bind=lambda model: bound.append(model))
    with pytest.raises(NotImplementedError, match="directed"):
        NativeIdentityFlavorProgram(relation, source, directed)
    assert bound == []
    with pytest.raises(NotImplementedError, match="directed"):
        relation.bind(source, directed=directed)
    ordinary = relation.bind(source)
    for method in (
        ordinary.evaluate_directed_native,
        ordinary.evaluate_directed_decimal,
    ):
        with pytest.raises(ValueError, match="no retained directed"):
            method(_labelled_points(), ["1"] * 3)


def test_direct_constructor_and_rebind_require_same_prepared_owner(monkeypatch):
    relation, source, _, _ = _recording_source()
    foreign = NativeSourceModelProgram(
        source.source_bare, SimpleNamespace(prepared=object()), matched=()
    )
    with pytest.raises(ValueError, match="another prepared representative"):
        NativeIdentityFlavorProgram(relation, foreign)
    mapped = relation.bind(source)
    options = []

    def rebind(self, **kwargs):
        assert self is source
        options.append(kwargs)
        return foreign

    monkeypatch.setattr(NativeSourceModelProgram, "rebind", rebind)
    with pytest.raises(ValueError, match="another prepared representative"):
        mapped.rebind(arithmetic="arbitrary", precision=80)
    assert options == [{"arithmetic": "arbitrary", "precision": 80}]


def test_selected_ordinary_view_hides_only_unproved_features(monkeypatch):
    representative, _, calls = _directed(monkeypatch)
    original = representative._backend
    original._metadata["physics"]["resolved"] = True
    original._metadata["external_pdgs"] = [1, -1, 21, 21, 21]
    original._metadata["plan"].update(
        external_pdgs=[1, -1, 21, 21, 21],
        state_legs=[{"incoming": index < 2} for index in range(5)],
        helicities=[[-1, 1, -1, -1, -1], [-1, 1, 1, 1, 1]],
        spin_projections=[{"legs": [3]}],
    )
    original._metadata.update(
        physics_scope="selected-process",
        external_permutation=list(range(5)),
        representative_physics=deepcopy(original._metadata["physics"]),
    )
    preserved = deepcopy(original._metadata)
    selected = copy(original)
    selected._metadata = deepcopy(original._metadata)
    selected._metadata["physics"].update(resolved=False, correlations=False)
    selected._metadata["process_id"] = "bose-member"
    selected._metadata["external_permutation"] = [0, 1, 3, 4, 2]
    member = Runtime(selected)
    assert member.external_permutation == (0, 1, 3, 4, 2)
    assert representative.external_permutation == (0, 1, 2, 3, 4)
    info = member.inspect()
    assert info["physics_scope"] == "selected-process"
    assert not info["resolved"] and not info["correlations"]
    assert info["representative_physics"]["resolved"]
    assert info["representative_physics"]["correlations"]
    assert info["resolved_color_components"] == []
    assert info["color_correlations"] == []
    assert info["spin_correlations"] == ()
    assert member.physics.color_ids == ("full",)
    assert member.physics.born_anchored_color_components == ()
    with pytest.raises(CompatibilityError, match="resolved"):
        member.evaluate_resolved([], mu_r2=[])
    with pytest.raises(CompatibilityError, match="directed"):
        member.available_color_correlations()
    with pytest.raises(CompatibilityError, match="directed"):
        member.evaluate_correlated_many([], {"born": CorrelatedRequest()}, mu_r2=[])
    with pytest.raises(CompatibilityError, match="directed"):
        member.set_spin_correlation_vectors({3: [0, 1, 0, 0]})
    assert calls == []
    # Representative discovery still uses its complete saved catalogue.
    assert [item.id for item in representative.available_color_correlations()] == [
        "born",
        "x",
    ]
    assert original._spin_classes() == ((3,),)
    assert original._metadata == preserved
    assert member.evaluate([], mu_r2=[]).shape == (0, 3, 3)


@pytest.mark.parametrize("duplicate", [False, True])
def test_symmetric_field_orbit_is_still_required_exactly_once(field_control, duplicate):
    """Permutation invariance alone misses equally omitted/doubled field rows."""
    from pyamplicol.loops.flavor_transport import _bose_external_fields

    plan, fields, owner = field_control
    gluon = next(item for item in fields if item.external_index == 2)
    orbit = tuple(
        item
        for item in fields
        if item.born is gluon.born
        and item.rule == gluon.rule
        and item.alternative_index == gluon.alternative_index
        and item.external_index in (2, 3)
    )
    assert sorted(item.external_index for item in orbit) == [2, 3]
    changed = (
        (*fields, *orbit)
        if duplicate
        else tuple(item for item in fields if not any(item is x for x in orbit))
    )
    # This altered multiset remains exactly symmetric under 2<->3; only the
    # comparison with the complete selected source rules detects the error.
    with pytest.raises(ValueError, match="complete selected source rules"):
        _bose_external_fields(plan, owner(changed), (0, 1, 3, 2))
