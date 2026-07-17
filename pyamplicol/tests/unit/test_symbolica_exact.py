# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import ROUND_UP, Decimal, localcontext

import pytest

from pyamplicol.api.errors import CompatibilityError
from pyamplicol.runtime.symbolica_exact import (
    _apply_lc_replay_input_mapping,
    _apply_lc_replay_resolved,
    _ExactEvaluator,
    _lc_replay_plan,
    _upcast_decimal,
    _working_precision,
)


class _RecordingEvaluator:
    def __init__(self) -> None:
        self.values: object = None
        self.precision: int | None = None

    def evaluate_complex_with_prec(
        self, values: object, precision: int
    ) -> list[tuple[Decimal, Decimal]]:
        self.values = values
        self.precision = precision
        return [(Decimal("1.25"), Decimal("-0.5"))]


def test_upcast_decimal_preserves_value_and_carries_requested_precision() -> None:
    cases = (
        Decimal("500"),
        Decimal("0.00123"),
        Decimal("1e100"),
        Decimal("-7.25"),
    )

    for value in cases:
        upcast = _upcast_decimal(value, 40)
        assert upcast == value
        assert len(upcast.as_tuple().digits) == 40

    zero = _upcast_decimal(Decimal("0"), 40)
    assert zero == 0
    assert zero.as_tuple().exponent == -40


def test_exact_evaluator_upcasts_every_complex_input() -> None:
    recording = _RecordingEvaluator()
    evaluator = _ExactEvaluator((recording,))

    result = evaluator.evaluate(
        ((Decimal("500"), Decimal("0")), (Decimal("0.125"), Decimal("-2"))),
        80,
    )

    assert result == ((Decimal("1.25"), Decimal("-0.5")),)
    assert recording.precision == 80
    values = recording.values
    assert isinstance(values, tuple)
    for real, imaginary in values:
        assert isinstance(real, Decimal)
        assert isinstance(imaginary, Decimal)
        assert len(real.as_tuple().digits) == 80 or real.is_zero()
        assert len(imaginary.as_tuple().digits) == 80 or imaginary.is_zero()


def test_upcast_rounding_does_not_depend_on_ambient_decimal_context() -> None:
    with localcontext() as context:
        context.rounding = ROUND_UP
        assert _upcast_decimal(Decimal("1.25"), 2) == Decimal("1.2")


def test_every_exact_request_stays_above_symbolica_binary64_shortcut() -> None:
    assert all(_working_precision(precision) >= 40 for precision in range(1, 40))
    assert _working_precision(80) == 88


def _replay_metadata() -> tuple[dict[str, object], dict[str, object]]:
    execution: dict[str, object] = {
        "compiled": {
            "lc_topology_replay": {
                "enabled": True,
                "mode": "external-label-permutation",
                "replayed_sector_count": 2,
                "groups": [
                    {
                        "sector_permutations": [
                            {"weight": 2.0, "label_permutation": []},
                            {
                                "weight": 1.0,
                                "label_permutation": [
                                    {
                                        "representative_label": 3,
                                        "sector_label": 4,
                                    },
                                    {
                                        "representative_label": 4,
                                        "sector_label": 3,
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        }
    }
    physics: dict[str, object] = {
        "color_accuracy": "lc",
        "external_particles": [{}, {}, {}, {}],
        "helicities": [
            {
                "id": "h:+1,-1,+1,-1",
                "values": [1, -1, 1, -1],
                "coefficient": 1.0,
            },
            {
                "id": "h:+1,-1,-1,+1",
                "values": [1, -1, -1, 1],
                "coefficient": 1.0,
            },
        ],
        "color_components": [
            {
                "kind": "lc-flow",
                "id": "flow:1,2,3,4",
                "word": [1, 2, 3, 4],
                "coefficient": 1.0,
            },
            {
                "kind": "lc-flow",
                "id": "flow:1,4,3,2",
                "word": [1, 4, 3, 2],
                "coefficient": 1.0,
            },
            {
                "kind": "lc-flow",
                "id": "flow:1,2,4,3",
                "word": [1, 2, 4, 3],
                "coefficient": 1.0,
            },
        ],
        "reduction": {
            "groups": [
                {
                    "physical_helicity_ids": ["h:+1,-1,+1,-1"],
                    "physical_color_ids": ["flow:1,2,3,4"],
                }
            ]
        },
    }
    return execution, physics


def _materialized_cell(value: str) -> tuple[tuple[Decimal, ...], ...]:
    return (
        (Decimal(value), Decimal(0), Decimal(0)),
        (Decimal(0), Decimal(0), Decimal(0)),
    )


def test_exact_lc_replay_routes_public_flows_and_selectors() -> None:
    execution, physics = _replay_metadata()
    plan = _lc_replay_plan(execution, physics, None)
    assert plan is not None
    assert len(plan.entries) == 2
    materialized = (
        _materialized_cell("3"),
        _materialized_cell("5"),
        _materialized_cell("7"),
        _materialized_cell("11"),
    )
    helicity_ids = ("h:+1,-1,+1,-1", "h:+1,-1,-1,+1")
    color_ids = ("flow:1,2,3,4", "flow:1,4,3,2", "flow:1,2,4,3")
    full, _, _ = _apply_lc_replay_resolved(
        materialized,
        plan,
        2,
        helicity_ids,
        color_ids,
        None,
        None,
    )
    assert tuple(
        sum((value for helicity in point for value in helicity), Decimal(0))
        for point in full
    ) == (Decimal(13), Decimal(21))

    values, helicities, colors = _apply_lc_replay_resolved(
        materialized,
        plan,
        2,
        helicity_ids,
        color_ids,
        ("h:+1,-1,-1,+1",),
        ("flow:1,2,4,3",),
    )

    assert helicities == ("h:+1,-1,-1,+1",)
    assert colors == ("flow:1,2,4,3",)
    assert values == (((Decimal(7),),), ((Decimal(11),),))

    point = tuple(
        (Decimal(index), Decimal(0), Decimal(0), Decimal(0)) for index in range(4)
    )
    assert _apply_lc_replay_input_mapping(point, plan.entries[1].input_mapping) == (
        point[0],
        point[1],
        point[3],
        point[2],
    )


def test_exact_lc_replay_rejects_missing_public_flow_reduction() -> None:
    execution, physics = _replay_metadata()
    colors = list(physics["color_components"])  # type: ignore[arg-type]
    colors.pop(1)
    physics["color_components"] = colors

    with pytest.raises(CompatibilityError, match="missing replayed LC flow word"):
        _lc_replay_plan(execution, physics, None)
