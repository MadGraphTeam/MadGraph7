# SPDX-License-Identifier: 0BSD
"""Retained independent fixed-full Bose controls, with no graph construction.

Physical Bose equality is a completeness control, not an exact-leg oracle. The
separate distinguishable three-cycle transport test detects skipped/doubled
permutations, which these identical-boson amplitudes cannot distinguish.
"""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from .test_loop_physical_correlations import _close, _decimals

# Set only by the reviewed retained-owner runner after exact local admission.
RETAINED_BOSE = None


@pytest.fixture(scope="module")
def bose_pair():
    if RETAINED_BOSE is None:
        pytest.skip("requires the retained independently generated fixed-full pair")
    relation, sources = RETAINED_BOSE
    reference = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    )
    points = np.asarray(
        json.loads(reference.read_text(), parse_float=str, parse_int=str)["points"][:2],
        dtype=object,
    )
    return relation, sources, points


def _compare_complete(left, right, tolerance):
    assert left.orders == right.orders
    for name in (
        "born",
        "virtual",
        "bare",
        "uv",
        "r2",
        "uv_local",
        "uv_external_fields",
    ):
        _close(
            _decimals(getattr(left, name)), _decimals(getattr(right, name)), tolerance
        )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bits",
    [
        ("double", None, "fast", "3e-10", 20),
        ("double-double", None, "generic", "3e-24", 70),
        ("arbitrary", 70, "generic", "3e-54", 170),
    ],
)
def test_actual_species_and_bose_components_all_carriers(
    bose_pair, arithmetic, precision, engine, tolerance, bits
):
    relation, original, points = bose_pair
    original_snapshots = tuple(item.source_bare.state.snapshot() for item in original)
    bound = tuple(
        item.rebind(
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine=engine,
        )
        for item in original
    )
    for source, old, snapshot in zip(bound, original, original_snapshots, strict=True):
        assert source.prepared is old.prepared
        assert source.virtual.bare.backend is old.virtual.bare.backend
        assert source.source_bare.state is not old.source_bare.state
        assert old.source_bare.state.snapshot() is snapshot
        source.update_parameters({"aS": "0.118", "MT": "173", "MU_R": "1000"})
    mapped = relation.bind(bound[0])
    swapped = points[:, (0, 1, 3, 2), :]
    scales = ["1000000", "900000"]
    tolerance = Decimal(tolerance)
    with localcontext() as context:
        context.prec = 110
        values = [source.evaluate_ordered_decimal(points, scales) for source in bound]
        permuted = [
            source.evaluate_ordered_decimal(swapped, scales) for source in bound
        ]
        # Each species is checked against its own original, and the member was
        # generated independently. The transport's exact leg action is separate.
        _compare_complete(values[0], permuted[0], tolerance)
        _compare_complete(values[1], permuted[1], tolerance)
        _compare_complete(values[0], values[1], tolerance)
        _compare_complete(
            mapped.evaluate_ordered_decimal(points, scales), values[1], tolerance
        )
        shifted = str(Decimal("0.118") * (1 + Decimal(2) ** -bits))
        for source in bound:
            source.update_parameters({"aS": shifted})
        changed = mapped.evaluate_ordered_decimal(points[:1], ["2100000"])
        _compare_complete(
            changed,
            bound[1].evaluate_ordered_decimal(points[:1], ["2100000"]),
            tolerance,
        )
        assert tuple(changed.orders.virtual) == tuple(values[1].orders.virtual)
        assert any(
            a != b
            for a, b in zip(changed.born.flat, values[1].born[:1].flat, strict=True)
        )
        if arithmetic == "double":
            empty = mapped.evaluate_ordered_decimal(points[:0], [])
            assert empty.virtual.shape[0] == 0 and empty.born.shape[0] == 0
            reordered = points[[1, 0, 1]]
            _compare_complete(
                mapped.evaluate_ordered_decimal(reordered, ["9e5", "1e6", "9e5"]),
                bound[1].evaluate_ordered_decimal(reordered, ["9e5", "1e6", "9e5"]),
                tolerance,
            )
    for old, snapshot in zip(original, original_snapshots, strict=True):
        assert old.source_bare.state.snapshot() is snapshot
