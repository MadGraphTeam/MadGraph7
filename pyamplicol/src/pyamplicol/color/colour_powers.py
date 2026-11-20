# SPDX-License-Identifier: 0BSD
"""N_c-power bounds of colour factors, free of numerical dependencies.

Kept apart from :mod:`pyamplicol.color.adjoint_kernel`, which needs numpy, so
that the colour-factor algorithms (and the lightweight configuration API that
reaches them through the public imports) stay importable without it.
"""

from __future__ import annotations


def full_minimum_power(coloured_labels: int, full_col_acc: int | None) -> int:
    """The lowest N_c power a 'full' colour factor keeps: none dropped
    (``full_col_acc`` None, every caller's default), or the legacy depth of
    ``2 * full_col_acc`` powers below the leading one."""

    if full_col_acc is None:
        return 0
    return max(coloured_labels - 2 * full_col_acc, 0)
