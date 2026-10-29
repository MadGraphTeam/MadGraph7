# SPDX-License-Identifier: 0BSD
"""Typed native batch transport without decimal intermediates."""

import pytest
from _loop_reduction import NativeBatch

MODES = [("double", None), ("double-double", None), ("arbitrary", 80)]


@pytest.mark.parametrize("arithmetic,precision", MODES)
def test_native_nonfinite_scan_preserves_row_ids_without_numeric_conversion(
    arithmetic, precision
):
    batch = NativeBatch.from_decimal(
        [["1", "0"], ["2", "0"], ["NaN", "0"], ["0", "0"], ["1", "inf"], ["-inf", "0"]],
        3,
        2,
        arithmetic,
        precision,
        [91, 7, 42],
    )
    assert batch.nonfinite_rows() == [7, 42]
    assert batch.take_rows([2, 0, 1]).nonfinite_rows() == [42, 7]
    assert batch.take_rows([0]).nonfinite_rows() == []
    assert batch.take_rows([]).nonfinite_rows() == []
    assert batch.take_columns([]).nonfinite_rows() == []
