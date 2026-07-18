# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.generation.eager_tables import (
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
    pack_rows,
    unpack_rows,
)


@pytest.mark.parametrize(
    "rows",
    (
        (
            EagerInvocationRow(2, 3, 5, 7, 11, 13, 17, 19),
            EagerInvocationRow(23, 29, 31, 37, 41, 43, 47, 53),
        ),
        (
            EagerAttachmentRow(2, -0.5, 0.25),
            EagerAttachmentRow(3, 1.0, 0.0),
        ),
        (
            EagerCouplingRow(MISSING_U32, MISSING_U32, -0.5, 0.25),
            EagerCouplingRow(3, 5, 0.0, 0.0),
        ),
        (
            EagerFinalizationRow(MISSING_U32, 2, 3, MISSING_U32, 5),
            EagerFinalizationRow(7, 11, MISSING_U32, 13, 17),
        ),
        (
            EagerClosureRow(2, 3, 5, 7, 11, -1.0, 0.5),
            EagerClosureRow(13, 17, 19, 23, 29, 1.0, 0.0),
        ),
    ),
)
def test_eager_fixed_width_tables_round_trip(rows: tuple[object, ...]) -> None:
    row_type = type(rows[0])
    payload = pack_rows(rows)  # type: ignore[arg-type]
    assert len(payload) == len(rows) * row_type._STRUCT.size
    assert unpack_rows(payload, row_type) == rows


def test_eager_table_rejects_mixed_rows() -> None:
    with pytest.raises(TypeError, match="one row type"):
        pack_rows(
            (
                EagerAttachmentRow(0, 1.0, 0.0),
                EagerClosureRow(0, 0, 0, 0, 0, 1, 0),
            )
        )


def test_eager_table_rejects_truncated_payload() -> None:
    payload = pack_rows((EagerInvocationRow(0, 1, 2, 3, 4, 5, 6, 7),))
    with pytest.raises(ValueError, match="not a multiple"):
        unpack_rows(payload[:-1], EagerInvocationRow)


def test_eager_table_identifiers_are_bounded() -> None:
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        EagerAttachmentRow(1 << 32, 1.0, 0.0)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        EagerInvocationRow(0, 0, 0, 0, 0, 0, -1, 0)


@pytest.mark.parametrize(
    "row",
    (
        EagerAttachmentRow,
        EagerCouplingRow,
        EagerClosureRow,
    ),
)
def test_eager_table_floating_values_must_be_finite(row: type[object]) -> None:
    with pytest.raises(ValueError, match="finite"):
        if row is EagerAttachmentRow:
            EagerAttachmentRow(0, float("nan"), 0.0)
        elif row is EagerCouplingRow:
            EagerCouplingRow(MISSING_U32, MISSING_U32, 0.0, float("inf"))
        else:
            EagerClosureRow(0, 0, 0, 0, 0, float("-inf"), 0.0)
