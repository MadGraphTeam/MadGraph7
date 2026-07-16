# SPDX-License-Identifier: 0BSD
"""Construction of color sectors from canonical process data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import permutations, product

from ..processes import ProcessOptions
from ..processes.ir import CanonicalProcessIR, ProcessLegIR, build_process_ir
from .plan_types import (
    GenericColorPlan,
    LCColorSector,
    LCQuarkLine,
)


def build_color_plan(
    process: str | CanonicalProcessIR,
    *,
    color_accuracy: str = "lc",
    options: ProcessOptions | None = None,
    max_sectors: int | None = None,
    reference_color_order: Sequence[int] | None = None,
) -> GenericColorPlan:
    process_ir = (
        process
        if isinstance(process, CanonicalProcessIR)
        else build_process_ir(process, color_accuracy=color_accuracy, options=options)
    )
    max_sector_count = _normalize_sector_cap(max_sectors)
    if color_accuracy != process_ir.color_accuracy:
        process_ir = build_process_ir(
            process_ir.process,
            color_accuracy=color_accuracy,
            options=options,
        )
    quark_legs = _legs_by_labels(process_ir, process_ir.quark_labels)
    antiquark_legs = _legs_by_labels(process_ir, process_ir.antiquark_labels)
    gluon_labels = process_ir.gluon_labels
    singlet_labels = process_ir.singlet_labels
    if len(quark_legs) != len(antiquark_legs):
        return GenericColorPlan(
            process=process_ir,
            color_accuracy=color_accuracy,
            sectors=(),
            diagnostics=(
                "leading-colour open-line plan requires balanced outgoing "
                "quark and antiquark counts",
            ),
        )
    if not quark_legs:
        return _build_no_quark_color_plan(
            process_ir,
            gluon_labels=gluon_labels,
            singlet_labels=singlet_labels,
            max_sectors=max_sector_count,
            reference_color_order=reference_color_order,
        )

    sectors: list[LCColorSector] = list(
        _reference_lc_color_sectors(process_ir, reference_color_order)
    )
    seen_sector_keys = {_sector_dedup_key(sector) for sector in sectors}
    truncated = False
    for antiquark_permutation in permutations(antiquark_legs):
        for gluon_allocation in _iter_ordered_gluon_allocations(
            gluon_labels,
            len(quark_legs),
        ):
            lines = tuple(
                LCQuarkLine(
                    quark_label=quark.label,
                    antiquark_label=antiquark.label,
                    gluon_labels=tuple(gluon_allocation[index]),
                )
                for index, (quark, antiquark) in enumerate(
                    zip(quark_legs, antiquark_permutation, strict=True)
                )
            )
            for word_labels in _iter_open_line_color_words(
                lines,
                include_block_permutations=color_accuracy != "lc",
            ):
                candidate = LCColorSector(
                    id=len(sectors),
                    kind="open-lines",
                    quark_lines=lines,
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


def _build_no_quark_color_plan(
    process: CanonicalProcessIR,
    *,
    gluon_labels: tuple[int, ...],
    singlet_labels: tuple[int, ...],
    max_sectors: int | None,
    reference_color_order: Sequence[int] | None = None,
) -> GenericColorPlan:
    if not gluon_labels:
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

    first = min(gluon_labels)
    rest = tuple(label for label in gluon_labels if label != first)
    sectors: list[LCColorSector] = list(
        _reference_lc_color_sectors(process, reference_color_order)
    )
    seen_sector_keys = {_sector_dedup_key(sector) for sector in sectors}
    truncated = False
    fold_reflections = process.color_accuracy == "lc"
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
        *process.quark_labels,
        *process.antiquark_labels,
        *process.gluon_labels,
    }
    coloured_word = tuple(label for label in reference if label in coloured_labels)
    if sorted(coloured_word) != sorted(coloured_labels):
        return ()
    if not process.quark_labels:
        if (
            tuple(label for label in coloured_word if label in process.gluon_labels)
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
            quark_lines=lines,
            singlet_labels=process.singlet_labels,
            word_labels=word_labels,
        ),
    )


def _reference_open_lines(
    process: CanonicalProcessIR,
    coloured_word: tuple[int, ...],
    by_label: dict[int, ProcessLegIR],
) -> tuple[LCQuarkLine, ...] | None:
    quark_labels = set(process.quark_labels)
    antiquark_labels = set(process.antiquark_labels)
    gluon_labels = set(process.gluon_labels)
    lines: list[LCQuarkLine] = []
    offset = 0
    while offset < len(coloured_word):
        start = coloured_word[offset]
        if start in quark_labels:
            end_labels = antiquark_labels
            start_is_quark = True
        elif start in antiquark_labels:
            end_labels = quark_labels
            start_is_quark = False
        else:
            return None
        end = offset + 1
        while end < len(coloured_word) and coloured_word[end] in gluon_labels:
            end += 1
        if end >= len(coloured_word) or coloured_word[end] not in end_labels:
            return None
        middle = coloured_word[offset + 1 : end]
        if any(label not in gluon_labels for label in middle):
            return None
        stop = coloured_word[end]
        if start_is_quark:
            quark_label = start
            antiquark_label = stop
        else:
            quark_label = stop
            antiquark_label = start
        if by_label[quark_label].particle_class != "quark":
            return None
        if by_label[antiquark_label].particle_class != "antiquark":
            return None
        lines.append(
            LCQuarkLine(
                quark_label=quark_label,
                antiquark_label=antiquark_label,
                gluon_labels=middle,
            )
        )
        offset = end + 1
    if len(lines) != len(process.quark_labels):
        return None
    return tuple(lines)


def _sector_dedup_key(sector: LCColorSector) -> tuple[object, ...]:
    return (
        sector.kind,
        tuple(
            (
                line.quark_label,
                line.antiquark_label,
                line.gluon_labels,
                line.singlet_labels,
            )
            for line in sector.quark_lines
        ),
        sector.trace_labels,
        sector.singlet_labels,
        sector.word_labels,
    )


def _iter_open_line_color_words(
    lines: tuple[LCQuarkLine, ...],
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
    lines: tuple[LCQuarkLine, ...],
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
        line_by_coloured = {line.coloured_labels: line for line in sector.quark_lines}
        ordered_blocks = _ordered_open_line_blocks(
            sector.word_labels,
            sector.quark_lines,
        )
        if ordered_blocks is None:
            return (sector.word_labels,)
        ordered_lines = tuple(line_by_coloured[block] for block in ordered_blocks)
    else:
        ordered_lines = sector.quark_lines

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


def _legacy_line_orientations(line: LCQuarkLine) -> tuple[tuple[int, ...], ...]:
    canonical = line.line_labels
    antiquark_first = (
        line.antiquark_label,
        *line.gluon_labels,
        line.quark_label,
        *line.singlet_labels,
    )
    if antiquark_first == canonical:
        return (canonical,)
    return (canonical, antiquark_first)


def _iter_ordered_gluon_allocations(
    gluon_labels: tuple[int, ...],
    line_count: int,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    if line_count <= 0:
        yield ()
        return
    for ordered_gluons in permutations(gluon_labels):
        yield from _iter_split_ordered_sequence(tuple(ordered_gluons), line_count)


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
