# SPDX-License-Identifier: 0BSD
"""Fresh saved/public EW transport over the independently checked source plans."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .test_loop_ew_directed_physical import COMPONENTS, TERMS, _check_complete
from .test_loop_native_directed import _blocks, _values
from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _decimals
from .test_loop_source_portable import _forbid_source


@pytest.fixture(scope="module")
def ew_runtime():
    root = os.environ.get("PYAMPLICOL_EW_COLOR_FIXTURES")
    if not root:
        pytest.skip("actual independently checked EW saved plans required")
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    verify_native_module(_rusticol)
    assert load_loop_native() is _rusticol._loop
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/real_mass_ew_quarks.json"
    )
    return Path(root), _rusticol._loop, json.loads(path.read_text())


def _frames(result, born_orders, virtual_orders, histories=None):
    rows = len(result.point_ids)
    prefix = (rows,) if histories is None else (rows, histories)
    values = dict(
        born=_values(result.born, (*prefix, born_orders)),
        **{
            name: _values(batch, (*prefix, virtual_orders, 3))
            for name, batch in zip(
                COMPONENTS[1:], result.component_batches, strict=True
            )
        },
    )
    if histories is not None:
        values.update(
            first=_values(result.directed_first, (*prefix, virtual_orders, 3)),
            second=_values(result.directed_second, (*prefix, virtual_orders, 3)),
        )
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "3e-9", 20),
        ("double-double", None, "3e-24", 70),
        ("arbitrary", 70, "3e-54", 170),
    ),
)
def test_saved_ew_complete_components_and_both_terms(
    ew_runtime, monkeypatch, arithmetic, precision, tolerance, bit
):
    root, native, fixture = ew_runtime
    _forbid_source(monkeypatch)
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
    )
    ordinary = native.NativeLoopProgram(
        str(root / "eeuu-qed-ordinary.pacbin"), **options
    )
    directed = native.NativeLoopProgram(
        str(root / "eeuu-qed-directed.pacbin"), **options
    )
    metadata = json.loads(directed.metadata_json())
    orders = metadata["order_layout"]
    assert orders["correction_names"] == ["QED"]
    b, v = len(orders["born"]), len(orders["virtual"])
    histories = metadata["directed_projection"]["histories"]
    # These factors were independently derived from this actual fixture's
    # dependency charge tensors in the source-level control, not a runtime rule.
    factors = ("1", "4/3", "-4/3", "-4/3", "4/3")
    assert len(histories) == len(factors)
    points, scales = fixture["points"], fixture["mu_r2"]
    ids = [2**64 - 1, 7, 42, 99]
    original_snapshot = directed.source_state.snapshot()
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        plain = ordinary.evaluate(points, scales, point_ids=ids)
        result = directed.evaluate_directed(points, scales, point_ids=ids)
        assert result.point_ids == result.requested_point_ids == ids
        assert result.history_ids == [item["id"] for item in histories]
        assert result.spin_legs == []
        assert result.observable_kind == "directed-complex"
        assert not result.failures
        before = _frames(result, b, v, len(histories))
        baseline = _check_complete(before, _frames(plain, b, v), factors, tolerance)
        # All three carriers are compared to the same binary64 provider fixture.
        # The much tighter charge/component identities above are internal checks.
        for name in COMPONENTS:
            expected = np.asarray(
                [row[name] for row in fixture["madloop"]["rows"]], dtype=object
            )
            expected = expected[:, None] if name == "born" else expected[:, None, :]
            paired = np.stack(
                (expected, np.zeros(expected.shape, dtype=object)), axis=-1
            )
            _close_component(baseline[name][:, 0], _decimals(paired), Decimal("3e-9"))

        gf = Decimal(fixture["parameters"]["Gf"])
        shifted = str(gf * (1 + Decimal(2) ** -bit))
        ordinary.source_state.update({"Gf": (shifted, "0")})
        changed_snapshot = directed.source_state.update({"Gf": (shifted, "0")})
        assert tuple(changed_snapshot.overrides["Gf"]) == (shifted, "0")
        rows = [2, 0, 2]
        selected = [points[i] for i in rows]
        selected_scales = [scales[i] for i in rows]
        changed = directed.evaluate_directed(
            selected, selected_scales, point_ids=ids[:3]
        )
        updated = _check_complete(
            _frames(changed, b, v, len(histories)),
            _frames(ordinary.evaluate(selected, selected_scales), b, v),
            factors,
            tolerance,
        )
        assert np.any(updated["born"][1] != baseline["born"][0])
        for name in (*COMPONENTS, *TERMS):
            assert np.array_equal(updated[name][0], updated[name][2])
        replay = directed.evaluate_directed(
            points[:1], scales[:1], point_ids=ids[:1], snapshot=original_snapshot
        )
        assert (
            replay.directed_first.to_decimal()
            == result.directed_first.take_rows([0]).to_decimal()
        )
        assert directed.source_state.snapshot().revision == changed_snapshot.revision
        if arithmetic == "double":
            empty = directed.evaluate_directed([], [], point_ids=[])
            assert empty.history_ids == result.history_ids and empty.point_ids == []
            assert all(batch.rows == 0 for batch in _blocks(empty))
            again = directed.evaluate_directed(
                selected[:1], selected_scales[:1], point_ids=[91]
            )
            for name in (*COMPONENTS, *TERMS):
                _close_component(
                    getattr(_frames(again, b, v, len(histories)), name),
                    updated[name][:1],
                    tolerance,
                )


def test_public_ew_metadata_grouped_charges_and_subset_rescue(ew_runtime):
    from pyamplicol import CorrelatedRequest, Runtime
    from pyamplicol.api import VirtualArithmeticSpec, VirtualStabilityPolicy
    from pyamplicol.api.errors import EvaluationError

    root, _, fixture = ew_runtime
    runtime = Runtime.load(root / "artifact")
    inspection = runtime.inspect()
    assert inspection["interference"] == "Bdagger K_BL L + Ldagger K_LB B"
    assert tuple(inspection["orders"]["correction_names"]) == ("QED",)
    assert runtime.physics.reduction.kind == "contracted-color"
    assert inspection["spin_correlations"] == ()
    names = {item.id for item in runtime.available_color_correlations()}
    assert {"born", *(f"history-{i}" for i in range(5))} <= names
    requests = {
        "identity": CorrelatedRequest("born"),
        "casimir": CorrelatedRequest("history-1"),
        "charge": CorrelatedRequest("history-2"),
    }
    points = [fixture["points"][0]] * 3
    scales = ["1000000", "1e-400", "1000000"]
    with pytest.raises(EvaluationError) as failure:
        runtime.evaluate_correlated_many(points, requests, mu_r2=scales)
    assert {item.point_id for item in failure.value.failures} == {1}
    results = runtime.evaluate_correlated_many(
        points,
        requests,
        mu_r2=scales,
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
        assert result.shape == (3, 1, 3)
        assert result.epsilon_powers == (0, -1, -2)
        assert result.correlated
        assert result.born_orders == ((("QED", 4),),)
        assert result.sectors[0].order == (("QED", 6),)
        assert [row.status for row in result.diagnostics] == [
            "success",
            "rescued",
            "success",
        ]
        assert result.values[0] == result.values[2]
    for name, factor in (("casimir", 4 / 3), ("charge", -4 / 3)):
        assert np.allclose(
            np.asarray(results[name].values, dtype=complex),
            factor * np.asarray(results["identity"].values, dtype=complex),
            rtol=1e-8,
            atol=1e-10,
        )
    before = runtime.evaluate_correlated(
        points[:1], color_correlation="history-2", mu_r2=scales[:1]
    )
    runtime.clear()
    assert (
        runtime.evaluate_correlated(
            points[:1], color_correlation="history-2", mu_r2=scales[:1]
        ).values
        == before.values
    )
    assert runtime.evaluate_correlated(
        [], color_correlation="history-1", mu_r2=[]
    ).shape == (0, 1, 3)
