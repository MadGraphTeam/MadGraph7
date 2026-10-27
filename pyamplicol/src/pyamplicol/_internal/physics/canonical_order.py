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

CANONICAL_NAMESPACE = "pyamplicol_canonical"
# Tensor heads (``spenso::tensor`` tag), for tensor networks on canonical
# symbols (``CanonicalRenaming(tensor_heads=...)``).
CANONICAL_TENSOR_NAMESPACE = "pyamplicol_canonical_tensor"
_BLOCK = 1 << 16
_POOL: list[Any] = []
_TENSOR_POOL: list[Any] = []
_POOL_LOCK = RLock()
# Block offsets whose relative order changes under every misalignment of the
# block by 1 to 2^16 - 1 ids (both byte boundaries and the block's end).
_ALIGNMENT_PROBE = (0, 1, 255, 256, 257, 511, 65280, 65535)
_POOL_TOKEN = re.compile(
    rf"(?<![\w:])({CANONICAL_NAMESPACE}|{CANONICAL_TENSOR_NAMESPACE})"
    r"::(?:\{[^{}]*\}::)?c([0-9]+)(?!\w)"
)
_NATURAL = re.compile(r"([0-9]+)")
# A binary64 literal of a canonical string (symbol names have no dots).
_FLOAT = re.compile(r"[0-9]\.[0-9]")
_TENSOR_TAG = "{spenso::tensor}"
# A symbol of a canonical string: ``namespace(::namespace)*::{attributes}::name``,
# or a bare built-in name (``conj``). Any other word it matches (the imaginary
# unit) is in no renaming and stays as it is.
_SYMBOL_TOKEN = re.compile(
    r"(?<![\w:])(?:(?:[^\W\d]\w*::)+\{[^{}]*\}::)?[^\W\d]\w*(?!\w)"
)


def canonical_symbols(count: int) -> list[Any]:
    """The first ``count`` canonical symbols, ordered alike in every process
    (module docstring). Each new block is checked (``_check_block``), so a
    misaligned block or a Symbolica whose order no longer follows the id
    bytes raises instead of silently making records depend on the process
    history again."""

    with _POOL_LOCK:
        while len(_POOL) < count:
            _new_blocks()
        return _POOL[:count]


def canonical_tensor_names(count: int) -> list[Any]:
    """The first ``count`` canonical tensor heads (Spenso ``TensorName``
    symbols, tagged ``spenso::tensor``), allocated like ``canonical_symbols``.

    Symbol ``k`` of either pool sorts by ``k``'s own bytes; a renaming gives
    every symbol a distinct ``k`` across both pools (``CanonicalRenaming``),
    so the two pools never tie. Within the first block only: beyond it two
    pools' blocks would order by their (history-dependent) positions."""

    if count > _BLOCK:
        raise ValueError(
            f"a canonical renaming with tensor heads is limited to {_BLOCK} symbols"
        )
    with _POOL_LOCK:
        if not _TENSOR_POOL:
            _new_blocks()
        return _TENSOR_POOL[:count]


def _new_blocks() -> None:
    """The next block of canonical symbols; the first time, the block of
    canonical tensor heads right after it (one padding for both; about 0.4 s
    once per process). Built through the central registry's lock, outside
    its cache."""

    from symbolica import Expression

    from .symbols import symbols

    def block(namespace: str, start: int, tags: list[str]) -> list[Any]:
        padding = -len(Expression.get_all_symbol_names()) % _BLOCK
        if padding:
            symbols.qualified(
                *(f"{namespace}::pad{start}_{k}" for k in range(padding)), tags=[]
            )
        created = list(
            symbols.qualified(
                *(f"{namespace}::c{start + k}" for k in range(_BLOCK)), tags=tags
            )
        )
        _check_block(created)
        for k in (0, _BLOCK - 1):
            token = _token(start + k, tensor=bool(tags))
            if created[k].to_canonical_string() != token:
                raise RuntimeError(
                    f"canonical symbol {created[k].to_canonical_string()} is not "
                    f"written {token}"
                )
        return created

    _POOL.extend(block(CANONICAL_NAMESPACE, len(_POOL), []))
    if not _TENSOR_POOL:
        _TENSOR_POOL.extend(block(CANONICAL_TENSOR_NAMESPACE, 0, ["spenso::tensor"]))


def _token(k: int, *, tensor: bool) -> str:
    """The canonical string of canonical symbol ``k`` (checked on every
    block)."""

    if tensor:
        return f"{CANONICAL_TENSOR_NAMESPACE}::{_TENSOR_TAG}::c{k}"
    return f"{CANONICAL_NAMESPACE}::{{}}::c{k}"


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
        tuple(
            (0, int(part)) if index % 2 else (1, part)
            for index, part in enumerate(parts)
        ),
        token,
    )


class CanonicalRenaming:
    """The canonical renaming of a set of symbols: each renamed symbol, in
    the natural order of the canonical names, gets the canonical symbol of
    its index ``k`` (a tensor head, ``tensor_heads``, the canonical tensor
    head ``k``). Every other symbol of a renamed text keeps its name."""

    def __init__(
        self,
        renamed: Iterable[Any],
        *,
        tensor_heads: Iterable[Any] = (),
    ) -> None:
        tokens = {symbol.to_canonical_string(): symbol for symbol in renamed}
        heads = {symbol.to_canonical_string() for symbol in tensor_heads}
        unknown = heads - tokens.keys()
        if unknown:
            raise ValueError(f"tensor heads not renamed: {sorted(unknown)}")
        for token in heads:
            if f"::{_TENSOR_TAG}::" not in token:
                raise ValueError(f"not a plain Spenso tensor head: {token}")
        for token in tokens:
            if not _SYMBOL_TOKEN.fullmatch(token):
                raise ValueError(f"unexpected canonical symbol token: {token}")
        self.renamed = sorted(tokens, key=_natural_key)
        self.index = {token: k for k, token in enumerate(self.renamed)}
        self.symbols = [tokens[token] for token in self.renamed]
        count = len(self.renamed)
        self.pool = canonical_symbols(count)
        tensors = canonical_tensor_names(count) if heads else []
        self.canonical_symbols = [
            tensors[k] if token in heads else self.pool[k]
            for k, token in enumerate(self.renamed)
        ]
        self._tensor = [token in heads for token in self.renamed]
        self._targets = {
            token: _token(k, tensor=tensor)
            for k, (token, tensor) in enumerate(
                zip(self.renamed, self._tensor, strict=True)
            )
        }

    def to_canonical(self, text: str) -> str:
        """A canonical string with the renamed symbols' canonical tokens.

        Every symbol of a canonical string is written in full
        (``namespace::{attributes}::name``), so each token is replaced or
        kept whole."""

        if not self._targets:
            return text
        targets = self._targets
        return _SYMBOL_TOKEN.sub(
            lambda match: targets.get(match.group(0), match.group(0)), text
        )

    def from_canonical(self, text: str, tokens: list[str] | None = None) -> str:
        """``text`` with canonical symbol ``k`` written as ``tokens[k]`` (by
        default the renamed symbol's canonical token)."""

        names = self.renamed if tokens is None else tokens
        return _POOL_TOKEN.sub(lambda match: names[int(match.group(2))], text)

    def canonical(self, expression: Any) -> Any:
        """``expression`` on the canonical symbols."""

        from symbolica import E

        return E(self.to_canonical(expression.to_canonical_string()))

    def symbol(self, original: Any) -> Any:
        """The canonical symbol of a renamed symbol."""

        return self.canonical_symbols[self.index[original.to_canonical_string()]]

    def name(self, original: Any) -> str:
        """The qualified name of a renamed symbol's canonical symbol (for a
        tensor head, the name of its ``TensorName``)."""

        k = self.index[original.to_canonical_string()]
        namespace = (
            CANONICAL_TENSOR_NAMESPACE if self._tensor[k] else CANONICAL_NAMESPACE
        )
        return f"{namespace}::c{k}"

    def restore(self, expression: Any) -> Any:
        """A canonical-symbol expression back on the original symbols."""

        from symbolica import E

        return E(self.from_canonical(expression.to_canonical_string()))


def _renaming(expression: Any, keep: Callable[[Any], bool] | None) -> CanonicalRenaming:
    every = expression.get_all_symbols(True)
    if keep is None:
        return CanonicalRenaming(every)
    return CanonicalRenaming(symbol for symbol in every if not keep(symbol))


def canonical_expression(
    expression: Any,
    *,
    keep: Callable[[Any], bool] | None = None,
    exact: bool = True,
) -> tuple[Any, CanonicalRenaming]:
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


def restore(expression: Any, renaming: CanonicalRenaming) -> Any:
    """A canonical-symbol expression back on the original symbols."""

    return renaming.restore(expression)


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
    memo: dict[str, Any] | None = None,
) -> Any:
    """``operation(expression)`` computed on canonical symbols (all of the
    expression's symbols but ``keep``'s) and returned on the original ones:
    an order-sensitive operation (``collect_horner``, ``collect_factors``)
    gives the same structure in every process.

    ``memo`` (owned by the caller, one per ``operation`` and ``keep``) keeps
    the operation's results by the canonical text of their argument: the
    result on canonical symbols is a function of that text, so expressions
    equal up to a renaming of their symbols in the same natural order share
    one operation."""

    from symbolica import E

    if memo is None:
        canonical, renaming = canonical_expression(expression, keep=keep)
        return restore(operation(canonical), renaming)
    text = expression.to_canonical_string()
    if _FLOAT.search(text):
        raise ValueError(
            "an expression with binary64 coefficients cannot be put in canonical "
            f"order exactly; write them as rationals: {text[:200]}"
        )
    renaming = _renaming(expression, keep)
    renamed = renaming.to_canonical(text)
    result = memo.get(renamed)
    if result is None:
        result = memo[renamed] = operation(E(renamed))
    return restore(result, renaming)


__all__ = [
    "CANONICAL_NAMESPACE",
    "CANONICAL_TENSOR_NAMESPACE",
    "CanonicalRenaming",
    "canonical_apply",
    "canonical_expression",
    "canonical_plain",
    "canonical_symbols",
    "canonical_tensor_names",
    "restore",
]
