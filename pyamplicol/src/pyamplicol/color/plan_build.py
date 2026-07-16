# SPDX-License-Identifier: 0BSD
"""Construction of color sectors from canonical process data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import permutations, product

from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from .plan_types import (
    GenericColorPlan,
    LCColorSector,
    LCOpenColorLine,
)


def build_color_plan(
    process: CanonicalProcessIR,
    *,
    color_accuracy: str = "lc",
    max_sectors: int | None = None,
    reference_color_order: Sequence[int] | None = None,
    fold_trace_reflections: bool = False,
) -> GenericColorPlan:
    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("color planning requires a model-resolved CanonicalProcessIR")
    process_ir = process
    max_sector_count = _normalize_sector_cap(max_sectors)
    if color_accuracy != process_ir.color_accuracy:
        raise ValueError("color plan accuracy must match the model-resolved process IR")
    fundamental_legs = _legs_by_labels(
        process_ir,
        process_ir.fundamental_labels,
    )
    antifundamental_legs = _legs_by_labels(
        process_ir,
        process_ir.antifundamental_labels,
    )
    adjoint_labels = process_ir.adjoint_labels
    singlet_labels = process_ir.singlet_labels
    if len(fundamental_legs) != len(antifundamental_legs):
        return GenericColorPlan(
            process=process_ir,
            color_accuracy=color_accuracy,
            sectors=(),
            diagnostics=(
                "leading-colour open-line plan requires balanced outgoing "
                "fundamental and antifundamental counts",
            ),
        )
    if not fundamental_legs:
        return _build_no_fundamental_color_plan(
            process_ir,
            adjoint_labels=adjoint_labels,
            singlet_labels=singlet_labels,
            max_sectors=max_sector_count,
            reference_color_order=reference_color_order,
            fold_trace_reflections=fold_trace_reflections,
        )

    sectors: list[LCColorSector] = list(
        _reference_lc_color_sectors(process_ir, reference_color_order)
    )
    seen_sector_keys = {_sector_dedup_key(sector) for sector in sectors}
    truncated = False
    for antifundamental_permutation in permutations(antifundamental_legs):
        for adjoint_allocation in _iter_ordered_adjoint_allocations(
            adjoint_labels,
            len(fundamental_legs),
        ):
            lines = tuple(
                LCOpenColorLine(
                    fundamental_label=fundamental.label,
                    antifundamental_label=antifundamental.label,
                    adjoint_labels=tuple(adjoint_allocation[index]),
                )
                for index, (fundamental, antifundamental) in enumerate(
                    zip(
                        fundamental_legs,
                        antifundamental_permutation,
                        strict=True,
                    )
                )
            )
            for word_labels in _iter_open_line_color_words(
                lines,
                include_block_permutations=color_accuracy != "lc",
            ):
                candidate = LCColorSector(
                    id=len(sectors),
                    kind="open-lines",
                    open_color_lines=lines,
                    singlet_labels=singlet_labels,
                    word_labels=word_labels,
                )
                key = _sector_dedup_key(candidate)
                if key in seen_sector_keys:
                    continue
                seen_sector_keys.add(key)
                sectors.append(candidate)
                if max_sector_count is not None and len(sectors) >= max_sector_count:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break

    diagnostics: tuple[str, ...] = ()
    if truncated:
        diagnostics = (
            f"leading-colour sector enumeration reached max_sectors={max_sector_count}",
        )
    return GenericColorPlan(
        process=process_ir,
        color_accuracy=color_accuracy,
        sectors=tuple(sectors),
        diagnostics=diagnostics,
        truncated=truncated,
    )


def _normalize_sector_cap(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    return None if normalized < 0 else normalized


def _build_no_fundamental_color_plan(
    process: CanonicalProcessIR,
    *,
    adjoint_labels: tuple[int, ...],
    singlet_labels: tuple[int, ...],
    max_sectors: int | None,
    reference_color_order: Sequence[int] | None = None,
    fold_trace_reflections: bool = False,
) -> GenericColorPlan:
    if not adjoint_labels:
        return GenericColorPlan(
            process=process,
            color_accuracy=process.color_accuracy,
            sectors=(
                LCColorSector(
                    id=0,
                    kind="singlet",
                    singlet_labels=singlet_labels,
                ),
            ),
        )

    first = min(adjoint_labels)
    rest = tuple(label for label in adjoint_labels if label != first)
    sectors: list[LCColorSector] = list(
        _reference_lc_color_sectors(process, reference_color_order)
    )
    seen_sector_keys = {_sector_dedup_key(sector) for sector in sectors}
    truncated = False
    fold_reflections = process.color_accuracy == "lc" and fold_trace_reflections
    seen_reversal_classes: set[tuple[int, ...]] = set()
    for ordered_rest in permutations(rest):
        if fold_reflections:
            canonical = min(ordered_rest, tuple(reversed(ordered_rest)))
            if canonical in seen_reversal_classes:
                continue
            seen_reversal_classes.add(canonical)
        candidate = LCColorSector(
            id=len(sectors),
            kind="single-trace",
            trace_labels=(first, *ordered_rest),
            singlet_labels=singlet_labels,
        )
        key = _sector_dedup_key(candidate)
        if key in seen_sector_keys:
            continue
        seen_sector_keys.add(key)
        sectors.append(candidate)
        if max_sectors is not None and len(sectors) >= max_sectors:
            truncated = True
            break
    diagnostics: tuple[str, ...] = ()
    if truncated:
        diagnostics = (
            f"leading-colour trace enumeration reached max_sectors={max_sectors}",
        )
    return GenericColorPlan(
        process=process,
        color_accuracy=process.color_accuracy,
        sectors=tuple(sectors),
        diagnostics=diagnostics,
        truncated=truncated,
        trace_reflections_folded=fold_reflections,
    )


def _legs_by_labels(
    process: CanonicalProcessIR,
    labels: tuple[int, ...],
) -> tuple[ProcessLegIR, ...]:
    by_label = {leg.label: leg for leg in process.legs}
    return tuple(by_label[label] for label in labels)


def _reference_lc_color_sectors(
    process: CanonicalProcessIR,
    reference_color_order: Sequence[int] | None,
) -> tuple[LCColorSector, ...]:
    if reference_color_order is None:
        return ()
    reference = tuple(int(label) for label in reference_color_order)
    if not reference:
        return ()
    by_label = {leg.label: leg for leg in process.legs}
    coloured_labels = {
        *process.fundamental_labels,
        *process.antifundamental_labels,
        *process.adjoint_labels,
    }
    coloured_word = tuple(label for label in reference if label in coloured_labels)
    if sorted(coloured_word) != sorted(coloured_labels):
        return ()
    if not process.fundamental_labels:
        if (
            tuple(label for label in coloured_word if label in process.adjoint_labels)
            != coloured_word
        ):
            return ()
        return (
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=coloured_word,
                singlet_labels=process.singlet_labels,
            ),
        )
    lines = _reference_open_lines(process, coloured_word, by_label)
    if lines is None:
        return ()
    word_labels = tuple(label for line in lines for label in line.coloured_labels)
    return (
        LCColorSector(
            id=0,
            kind="open-lines",
            open_color_lines=lines,
            singlet_labels=process.singlet_labels,
            word_labels=word_labels,
        ),
    )


def _reference_open_lines(
    process: CanonicalProcessIR,
    coloured_word: tuple[int, ...],
    by_label: dict[int, ProcessLegIR],
) -> tuple[LCOpenColorLine, ...] | None:
    fundamental_labels = set(process.fundamental_labels)
    antifundamental_labels = set(process.antifundamental_labels)
    adjoint_labels = set(process.adjoint_labels)
    lines: list[LCOpenColorLine] = []
    offset = 0
    while offset < len(coloured_word):
        start = coloured_word[offset]
        if start in fundamental_labels:
            end_labels = antifundamental_labels
            start_is_fundamental = True
        elif start in antifundamental_labels:
            end_labels = fundamental_labels
            start_is_fundamental = False
        else:
            return None
        end = offset + 1
        while end < len(coloured_word) and coloured_word[end] in adjoint_labels:
            end += 1
        if end >= len(coloured_word) or coloured_word[end] not in end_labels:
            return None
        middle = coloured_word[offset + 1 : end]
        if any(label not in adjoint_labels for label in middle):
            return None
        stop = coloured_word[end]
        if start_is_fundamental:
            fundamental_label = start
            antifundamental_label = stop
        else:
            fundamental_label = stop
            antifundamental_label = start
        if by_label[fundamental_label].color_role != "fundamental":
            return None
        if by_label[antifundamental_label].color_role != "antifundamental":
            return None
        lines.append(
            LCOpenColorLine(
                fundamental_label=fundamental_label,
                antifundamental_label=antifundamental_label,
                adjoint_labels=middle,
            )
        )
        offset = end + 1
    if len(lines) != len(process.fundamental_labels):
        return None
    return tuple(lines)


def _sector_dedup_key(sector: LCColorSector) -> tuple[object, ...]:
    return (
        sector.kind,
        tuple(
            (
                line.fundamental_label,
                line.antifundamental_label,
                line.adjoint_labels,
                line.singlet_labels,
            )
            for line in sector.open_color_lines
        ),
        sector.trace_labels,
        sector.singlet_labels,
        sector.word_labels,
    )


def _iter_open_line_color_words(
    lines: tuple[LCOpenColorLine, ...],
    *,
    include_block_permutations: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Return explicit colour words for one open-line pairing/allocation."""

    if not include_block_permutations or len(lines) < 2:
        return (tuple(label for line in lines for label in line.coloured_labels),)
    words: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for line_permutation in permutations(lines):
        word = tuple(
            label for line in line_permutation for label in line.coloured_labels
        )
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
    return tuple(words)


def _ordered_open_line_blocks(
    word: tuple[int, ...],
    lines: tuple[LCOpenColorLine, ...],
) -> tuple[tuple[int, ...], ...] | None:
    remaining = list(lines)
    blocks: list[tuple[int, ...]] = []
    offset = 0
    while offset < len(word):
        matched_index = None
        for index, line in enumerate(remaining):
            block = line.coloured_labels
            if word[offset : offset + len(block)] == block:
                matched_index = index
                blocks.append(block)
                offset += len(block)
                break
        if matched_index is None:
            return None
        remaining.pop(matched_index)
    if remaining:
        return None
    return tuple(blocks)


def _open_line_legacy_order_words(
    sector: LCColorSector,
) -> tuple[tuple[int, ...], ...]:
    """Full open-line block orders accepted in legacy process rows."""

    if sector.word_labels:
        line_by_coloured = {
            line.coloured_labels: line for line in sector.open_color_lines
        }
        ordered_blocks = _ordered_open_line_blocks(
            sector.word_labels,
            sector.open_color_lines,
        )
        if ordered_blocks is None:
            return (sector.word_labels,)
        ordered_lines = tuple(line_by_coloured[block] for block in ordered_blocks)
    else:
        ordered_lines = sector.open_color_lines

    line_orientations = tuple(_legacy_line_orientations(line) for line in ordered_lines)
    words: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for line_permutation in permutations(range(len(ordered_lines))):
        orientation_choices = (line_orientations[index] for index in line_permutation)
        for blocks in product(*orientation_choices):
            word = tuple(label for block in blocks for label in block)
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
    return tuple(words)


def _legacy_line_orientations(
    line: LCOpenColorLine,
) -> tuple[tuple[int, ...], ...]:
    canonical = line.line_labels
    antifundamental_first = (
        line.antifundamental_label,
        *line.adjoint_labels,
        line.fundamental_label,
        *line.singlet_labels,
    )
    if antifundamental_first == canonical:
        return (canonical,)
    return (canonical, antifundamental_first)


def _iter_ordered_adjoint_allocations(
    adjoint_labels: tuple[int, ...],
    line_count: int,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    if line_count <= 0:
        yield ()
        return
    for ordered_adjoint_labels in permutations(adjoint_labels):
        yield from _iter_split_ordered_sequence(
            tuple(ordered_adjoint_labels),
            line_count,
        )


def _iter_ordered_label_allocations(
    labels: tuple[int, ...],
    line_count: int,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    if line_count <= 0:
        yield ()
        return
    if not labels:
        yield tuple(() for _ in range(line_count))
        return
    buckets: list[list[int]] = [[] for _ in range(line_count)]
    yield from _assign_unordered_labels_to_buckets(labels, buckets, 0)


def _assign_unordered_labels_to_buckets(
    labels: tuple[int, ...],
    buckets: list[list[int]],
    index: int,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    if index == len(labels):
        yield tuple(tuple(bucket) for bucket in buckets)
        return
    label = labels[index]
    for bucket in buckets:
        bucket.append(label)
        yield from _assign_unordered_labels_to_buckets(labels, buckets, index + 1)
        bucket.pop()


def _iter_split_ordered_sequence(
    sequence: tuple[int, ...],
    bin_count: int,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    if bin_count == 1:
        yield (sequence,)
        return
    for split_index in range(len(sequence) + 1):
        head = sequence[:split_index]
        for tail in _iter_split_ordered_sequence(
            sequence[split_index:],
            bin_count - 1,
        ):
            yield (head, *tail)
