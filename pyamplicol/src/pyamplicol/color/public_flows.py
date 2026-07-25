# SPDX-License-Identifier: 0BSD
"""Authenticated compatibility mappings for public LC-flow selectors."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def amplicol_legacy_two_line_public_word(
    construction_word: Sequence[int],
    open_lines: Sequence[tuple[int, int, Sequence[int], Sequence[int]]],
    initial_source_slots: Collection[int],
) -> tuple[int, ...]:
    """Return the legacy AmpliCol spelling of a crossed two-line LC flow.

    Recurrence construction orders an open line from its fundamental endpoint
    to its antifundamental endpoint.  For the crossed two-quark-line sector,
    legacy AmpliCol instead exposes the final-to-initial line first.  This
    mapping is admitted only when the complete two-line topology proves that
    exact relation; all other words are returned unchanged.
    """

    word = tuple(int(slot) for slot in construction_word)
    if len(open_lines) != 2:
        return word
    canonical_lines = tuple(
        (
            int(fundamental),
            int(antifundamental),
            tuple(int(slot) for slot in adjoints),
            tuple(int(slot) for slot in singlets),
        )
        for fundamental, antifundamental, adjoints, singlets in open_lines
    )
    if any(adjoint or singlet for _, _, adjoint, singlet in canonical_lines):
        return word
    endpoints = tuple(
        slot
        for fundamental, antifundamental, _, _ in canonical_lines
        for slot in (fundamental, antifundamental)
    )
    if len(set(endpoints)) != 4:
        return word

    initial = {int(slot) for slot in initial_source_slots}
    initial_to_final = tuple(
        line
        for line in canonical_lines
        if line[0] in initial and line[1] not in initial
    )
    final_to_initial = tuple(
        line
        for line in canonical_lines
        if line[0] not in initial and line[1] in initial
    )
    if len(initial_to_final) != 1 or len(final_to_initial) != 1:
        return word
    incoming = initial_to_final[0]
    outgoing = final_to_initial[0]
    if word != (incoming[0], incoming[1], outgoing[0], outgoing[1]):
        return word
    return (outgoing[0], outgoing[1], incoming[0], incoming[1])


__all__ = ["amplicol_legacy_two_line_public_word"]
