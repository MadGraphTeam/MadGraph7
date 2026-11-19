# SPDX-License-Identifier: 0BSD
"""Pre-encoded JSON fragments: a large table (a profile's per-sample rows)
encoded once, compactly and with progress, inside the command's monitored
phases, then spliced verbatim into the ``--json`` document or a records
file (``cli.main.write_result``, ``splice``)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawJson:
    """A JSON value already encoded (``text``), of ``rows`` rows."""

    text: str
    rows: int = 0


def encode_rows(
    rows: Iterable[Sequence[object]],
    total: int,
    progress: Callable[[int], None] | None = None,
    every: int = 5000,
) -> RawJson:
    """``rows`` as one JSON array, one compact row per line (non-finite
    numbers must already be text); ``progress(done)`` every ``every``
    rows."""

    parts = []
    done = 0
    for row in rows:
        parts.append(json.dumps(row, separators=(",", ":"), allow_nan=False))
        done += 1
        if progress is not None and done % every == 0:
            progress(done)
    if progress is not None:
        progress(done)
    text = "[]" if not parts else "[\n" + ",\n".join(parts) + "\n]"
    return RawJson(text, done)


def placeholder(index: int) -> str:
    """The string a ``RawJson`` stands as in a document before ``splice``."""

    return f"\x00raw-json-{index}\x00"


def splice(encoded: str, fragments: Sequence[RawJson]) -> str:
    """``encoded`` (a ``json.dumps`` text) with every fragment's placeholder
    string replaced by the fragment's own text."""

    for index, fragment in enumerate(fragments):
        token = json.dumps(placeholder(index))
        encoded = encoded.replace(token, fragment.text, 1)
    return encoded


__all__ = ["RawJson", "encode_rows", "placeholder", "splice"]
