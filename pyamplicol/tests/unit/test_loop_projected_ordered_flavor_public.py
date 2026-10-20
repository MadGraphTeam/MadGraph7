# SPDX-License-Identifier: 0BSD
"""Public projected member totals over actual normally published checkpoints.

The public API exposes Born and complete complex totals. Separate integrated
terms and local components are covered by the saved-native controls.
"""

import os
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from .test_loop_generation_spin_flavor_runtime import _vectors
from .test_loop_ordered_flavor_physical import _points
from .test_loop_projected_flavor_runtime import _close_values

HISTORIES = ("born", "casimir", "ordered", "swapped")


@pytest.fixture(scope="module")
def published():
    root = os.environ.get("PYAMPLICOL_PROJECTED_FLAVOR_FIXTURES")
    if not root:
        pytest.skip("requires actual published projected ordered flavour artifacts")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module

    verify_native_module(_rusticol)
    return Path(root), _points()


def _conjugate(values):
    from pyamplicol.correlators import CorrelatedValue

    if isinstance(values, tuple):
        return tuple(_conjugate(value) for value in values)
    if isinstance(values, complex):
        return values.conjugate()
    assert isinstance(values, CorrelatedValue)
    return CorrelatedValue(values.real, values.imag.copy_negate())


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_public_projected_member_histories_sources_and_complex_totals(
    published, accuracy
):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api.errors import CompatibilityError

    root, points = published
    parameters = {"aS": "0.118", "MB": "0", "ymb": "0", "MT": "173", "MU_R": "1000"}
    member = Runtime.load(
        root / f"{accuracy}-mapped-artifact",
        process="ssgg",
        model_parameters=parameters,
    )
    independent = Runtime.load(
        root / f"{accuracy}-independent-artifact",
        process="ssgg",
        model_parameters=parameters,
    )
    representative = Runtime.load(root / f"{accuracy}-mapped-artifact", process="ddgg")
    info = member.inspect()
    assert info["process_id"] == "ssgg"
    assert representative.inspect()["process_id"] == "ddgg"
    assert info["color_accuracy"] == accuracy
    assert info["correlation_semantics"] == (
        "Bdagger K_BL L + Ldagger K_LB B; ordered histories may be complex"
    )
    assert info["spin_correlations"] == ((3,), (3, 4), (4,))
    assert info["resolved_color_components"] == []
    assert info["resolved_color_semantics"] == "one contracted colour component"
    catalogues = [
        runtime.available_color_correlations()
        for runtime in (member, independent, representative)
    ]
    assert all(
        tuple(item.id for item in catalog) == HISTORIES for catalog in catalogues
    )
    assert catalogues[0] == catalogues[1] == catalogues[2]
    with pytest.raises(CompatibilityError, match="resolved"):
        member.evaluate_resolved(points[:1], mu_r2="1000000")

    a, b = _vectors()
    classes = {"ordinary": {}, "left": {3: a}, "right": {4: b}, "joint": {3: a, 4: b}}
    requests = {
        f"{label}:{history}": CorrelatedRequest(history, vectors)
        for label, vectors in classes.items()
        for history in HISTORIES
    }
    batch, scales = [points[1], points[0]], ["900000", "1000000"]
    actual = member.evaluate_correlated_many(batch, requests, mu_r2=scales)
    expected = independent.evaluate_correlated_many(batch, requests, mu_r2=scales)
    assert tuple(actual) == tuple(requests)
    for label in requests:
        result, reference = actual[label], expected[label]
        assert result.correlated and result.color_accuracy == accuracy
        assert result.shape == reference.shape == (2, len(result.sectors), 3)
        assert result.born_shape == (2, len(result.born_orders))
        assert result.epsilon_powers == (0, -1, -2)
        assert result.born_orders == reference.born_orders
        assert result.sectors == reference.sectors
        assert result.valid == (True, True)
        assert all(
            item.status == "success" and not item.failures
            for item in result.diagnostics
        )
        _close_values(result.born_values, reference.born_values)
        _close_values(result.values, reference.values)
    for label in classes:
        ordered, swapped = actual[f"{label}:ordered"], actual[f"{label}:swapped"]
        _close_values(ordered.values, _conjugate(swapped.values))
    if accuracy == "nlc":
        # An unconditional 2Re presentation would erase this actual witness.
        assert any(
            abs(value.imag) > 1e-6
            for point in actual["ordinary:ordered"].values
            for sector in point
            for value in sector
        )
    empty = member.evaluate_correlated_many(
        [], {"joint": CorrelatedRequest("ordered", {3: a, 4: b})}, mu_r2=[]
    )["joint"]
    assert empty.shape == (0, len(actual["joint:ordered"].sectors), 3)
    assert empty.values == empty.born_values == empty.diagnostics == ()


def test_public_projected_ordered_member_subset_rescue_preserves_positions(published):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy

    root, points = published
    runtime = Runtime.load(root / "nlc-mapped-artifact", process="ssgg")
    a, b = _vectors()
    with localcontext() as context:
        context.prec = 100
        results = runtime.evaluate_correlated_many(
            [points[0]] * 3,
            {
                name: CorrelatedRequest(name, {3: a, 4: b})
                for name in ("ordered", "swapped")
            },
            mu_r2=["1000000", "1e-400", "1000000"],
            stability=VirtualStabilityPolicy(
                (
                    VirtualArithmeticSpec("arbitrary", 60),
                    VirtualArithmeticSpec("arbitrary", 80),
                ),
                "1e-8",
                "1e-9",
            ),
        )
    for result in results.values():
        assert [row.status for row in result.diagnostics] == [
            "success",
            "rescued",
            "success",
        ]
        assert result.valid == (True, True, True)
        assert result.values[0] == result.values[2]
        assert result.born_values[0] == result.born_values[2]
        verification = result.diagnostics[1].verification
        assert verification.point_id == 1
        assert verification.accuracy_established
        attempts = verification.attempts
        assert {
            failure.point_id for attempt in attempts for failure in attempt.failures
        } == {1}
        assert [attempt.arithmetic.arithmetic for attempt in attempts][-2:] == [
            "arbitrary",
            "arbitrary",
        ]
        assert all(
            failure.history_index is None or 0 <= failure.history_index < len(HISTORIES)
            for attempt in attempts
            for failure in attempt.failures
        )
    # The public API labels event positions; native and SDK tests cover opaque IDs.
    _close_values(
        results["ordered"].values,
        _conjugate(results["swapped"].values),
        Decimal("3e-9"),
    )
