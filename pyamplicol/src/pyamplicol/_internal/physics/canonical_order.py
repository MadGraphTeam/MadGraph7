# SPDX-License-Identifier: 0BSD
"""Symbolica normal forms that do not depend on the process history.

Symbolica numbers symbols in creation order and orders the terms of a sum,
the factors of a product and the arguments of a symmetric or antisymmetric
function by the packed little-endian bytes of the symbols' ids
(``AtomView::cmp``; one byte below 256). The normal form of an expression,
and every result an order-sensitive operation derives from it (the printed
string, a Horner scheme, ``collect_factors``, the rounding of a binary64
evaluation), therefore depends on the absolute ids of its symbols: on every
symbol the interpreter created before (an earlier model compile, a tree
evaluation, a generation). The same model compiled in two processes gave
``f(c1,c2,c3)`` in one and ``f(c2,c3,c1)`` in the other.

``to_canonical_string`` prints an expression independently of the ids, but a
string in the plain format, a structural operation or an evaluation needs a
normal form. The functions here build it on canonical symbols: the
expression's symbols are renamed, in the (natural) order of their canonical
names, to the canonical symbols ``pyamplicol_canonical::c<k>``, whose ids are
allocated in whole blocks of 2^16 starting at a multiple of 2^16 (unrelated
padding symbols fill the gap), so that symbol ``k`` sorts by ``k``'s own
bytes in every process. The renaming goes through the id-independent
canonical string, so the canonical expression, and every result computed
from it, is the same in every process; it is exact (values do not change
beyond the rounding of a reordered evaluation).

The symbols a caller keeps (``keep``) are not renamed: Symbolica's and
Spenso's built-in heads, registered when Symbolica is imported, have the same
ids in every process.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from threading import RLock
from typing import Any

from .symbols import symbols

CANONICAL_NAMESPACE = "pyamplicol_canonical"
_BLOCK = 1 << 16
_POOL: list[Any] = []
_POOL_LOCK = RLock()
# Block offsets whose relative order changes under every misalignment of the
# block by 1 to 2^16 - 1 ids (both byte boundaries and the block's end).
_ALIGNMENT_PROBE = (0, 1, 255, 256, 257, 511, 65280, 65535)
_POOL_TOKEN = re.compile(
    rf"(?<![\w:]){CANONICAL_NAMESPACE}::(?:\{{\}}::)?c([0-9]+)(?!\w)"
)
_NATURAL = re.compile(r"([0-9]+)")
# A binary64 literal of a canonical string (symbol names have no dots).
_FLOAT = re.compile(r"[0-9]\.[0-9]")


def canonical_symbols(count: int) -> list[Any]:
    """The first ``count`` canonical symbols, ordered alike in every process
    (module docstring). A block costs about 0.2 s once per process; each new
    block is checked (``_check_block``), so a misaligned block or a Symbolica
    whose order no longer follows the id bytes raises instead of silently
    making records depend on the process history again."""

    from symbolica import Expression

    with _POOL_LOCK:
        while len(_POOL) < count:
            start = len(_POOL)
            padding = -len(Expression.get_all_symbol_names()) % _BLOCK
            if padding:
                symbols.qualified(
                    *(
                        f"{CANONICAL_NAMESPACE}::pad{start}_{k}"
                        for k in range(padding)
                    )
                )
            block = list(
                symbols.qualified(
                    *(f"{CANONICAL_NAMESPACE}::c{start + k}" for k in range(_BLOCK))
                )
            )
            _check_block(block)
            _POOL.extend(block)
        return _POOL[:count]


def _check_block(block: list[Any]) -> None:
    """Raise unless a sum of the probe symbols of a fresh block normalises in
    the order of their offsets' bytes (low byte first)."""

    from symbolica import Expression

    probe = sum((block[k] for k in _ALIGNMENT_PROBE), Expression.num(0))
    expected = sorted(_ALIGNMENT_PROBE, key=lambda k: (k % 256, k // 256))
    if [str(term) for term in probe] != [str(block[k]) for k in expected]:
        raise RuntimeError(
            "canonical Symbolica symbols are not ordered by their block offset "
            f"(got {probe}): the symbol ids are misaligned or Symbolica's order "
            "changed, so compiled records would depend on the process history"
        )


def _natural_key(token: str) -> tuple[object, ...]:
    parts = _NATURAL.split(token)
    return (
        tuple((0, int(part)) if index % 2 else (1, part) for index, part in enumerate(parts)),
        token,
    )


class _Renaming:
    """The canonical renaming of a set of symbols: canonical token ->
    canonical symbol index, in natural name order."""

    def __init__(self, renamed: Iterable[Any], kept: Iterable[Any]) -> None:
        tokens = {symbol.to_canonical_string(): symbol for symbol in renamed}
        self.renamed = sorted(tokens, key=_natural_key)
        self.index = {token: k for k, token in enumerate(self.renamed)}
        self.symbols = [tokens[token] for token in self.renamed]
        every = [*self.renamed, *(symbol.to_canonical_string() for symbol in kept)]
        # Longest first: a token never loses to one of its prefixes, and a
        # matched token (an attributed head's ``ns::{...}::name``) is consumed
        # whole, so nothing inside it is renamed.
        alternatives = "|".join(
            re.escape(token) for token in sorted(set(every), key=len, reverse=True)
        )
        self.pattern = re.compile(rf"(?<![\w:])(?:{alternatives})(?!\w)") if every else None
        self.pool = canonical_symbols(len(self.renamed))

    def to_canonical(self, text: str) -> str:
        if self.pattern is None:
            return text
        index = self.index
        return self.pattern.sub(
            lambda match: (
                f"{CANONICAL_NAMESPACE}::c{index[match.group(0)]}"
                if match.group(0) in index
                else match.group(0)
            ),
            text,
        )

    def from_canonical(self, text: str, tokens: list[str]) -> str:
        return _POOL_TOKEN.sub(lambda match: tokens[int(match.group(1))], text)


def _renaming(expression: Any, keep: Callable[[Any], bool] | None) -> _Renaming:
    every = expression.get_all_symbols(True)
    if keep is None:
        return _Renaming(every, ())
    return _Renaming(
        (symbol for symbol in every if not keep(symbol)),
        (symbol for symbol in every if keep(symbol)),
    )


def canonical_expression(
    expression: Any,
    *,
    keep: Callable[[Any], bool] | None = None,
    exact: bool = True,
) -> tuple[Any, _Renaming]:
    """``expression`` on canonical symbols (all of its symbols but ``keep``'s)
    and the renaming, for ``restore``.

    The renaming goes through the canonical string, which prints a binary64
    coefficient to 15 digits: ``exact`` refuses one rather than round it."""

    from symbolica import E

    text = expression.to_canonical_string()
    if exact and _FLOAT.search(text):
        raise ValueError(
            "an expression with binary64 coefficients cannot be put in canonical "
            f"order exactly; write them as rationals: {text[:200]}"
        )
    renaming = _renaming(expression, keep)
    return E(renaming.to_canonical(text)), renaming


def restore(expression: Any, renaming: _Renaming) -> Any:
    """A canonical-symbol expression back on the original symbols."""

    from symbolica import E

    return E(renaming.from_canonical(expression.to_canonical_string(), renaming.renamed))


def canonical_plain(expression: Any) -> str:
    """``expression.format_plain()`` in an order independent of the process
    history: the plain string of the expression's normal form on canonical
    symbols, with the original symbols' plain names. It parses back to
    ``expression``."""

    # A binary64 coefficient prints as ``format_plain`` prints it.
    canonical, renaming = canonical_expression(expression, exact=False)
    plain = [symbol.format_plain() for symbol in renaming.symbols]
    return renaming.from_canonical(canonical.format_plain(), plain)


def canonical_apply(
    operation: Callable[[Any], Any],
    expression: Any,
    *,
    keep: Callable[[Any], bool] | None = None,
) -> Any:
    """``operation(expression)`` computed on canonical symbols (all of the
    expression's symbols but ``keep``'s) and returned on the original ones:
    an order-sensitive operation (``collect_horner``, ``collect_factors``)
    gives the same structure in every process."""

    canonical, renaming = canonical_expression(expression, keep=keep)
    return restore(operation(canonical), renaming)


__all__ = [
    "CANONICAL_NAMESPACE",
    "canonical_apply",
    "canonical_expression",
    "canonical_plain",
    "canonical_symbols",
    "restore",
]
