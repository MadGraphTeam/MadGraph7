# SPDX-License-Identifier: 0BSD
"""Shared guard against silently promoting approximate symbolic constants."""

from __future__ import annotations

from typing import Any


def require_exact_numbers(expression: Any) -> None:
    """Reject approximate numeric atoms before evaluator rationalization."""
    from symbolica import AtomType, E, Expression

    if not isinstance(expression, Expression):
        raise TypeError("expected an exact Symbolica expression")
    pending = [expression]
    imaginary = E("1i")
    while pending:
        atom = pending.pop()
        if atom.get_type() == AtomType.Var:
            continue
        if atom.get_type() != AtomType.Num:
            pending.extend(atom)
            continue
        try:
            real = ((atom + atom.conj()) / 2).to_rational_polynomial().to_expression()
            imag = (
                ((atom - atom.conj()) / (2 * imaginary))
                .to_rational_polynomial()
                .to_expression()
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("exact expressions reject floating numeric atoms") from exc
        if not bool(atom.matches(real + imaginary * imag)):
            raise ValueError("exact expressions reject floating numeric atoms")
