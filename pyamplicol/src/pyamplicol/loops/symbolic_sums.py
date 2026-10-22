# SPDX-License-Identifier: 0BSD
"""Exact one-pass sums of many Symbolica expressions.

Python ``sum`` and repeated ``+=`` normalise the growing Symbolica sum again at
every step, so accumulating ``n`` terms costs ``O(n^2)``. Collect the terms of
each target in a list instead and add them once here: Symbolica's ``sum``
transformer normalises all arguments of one held function in a single pass.
The result is the same canonical expression as sequential addition.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Any

from .._internal.physics.symbols import symbols


@cache
def _sum_pattern():
    from symbolica import T

    # Private names: no generated expression can contain this head, so the
    # replacement only ever matches the one wrapper built below.
    head = symbols.qualified("pyamplicol_symbolic_sum::terms")
    terms = symbols.qualified("pyamplicol_symbolic_sum::terms__")
    return head, head(terms), terms.hold(T().sum())


def symbolic_sum(terms: Iterable[Any]) -> Any:
    """Return the exact sum of ``terms`` (Symbolica expressions or numbers)."""
    from symbolica import E

    terms = tuple(terms)
    if not terms:
        return E("0")
    head, pattern, total = _sum_pattern()
    return head(*terms).replace(pattern, total)
