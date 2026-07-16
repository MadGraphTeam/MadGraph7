# SPDX-License-Identifier: 0BSD
"""Legacy process-file parsing, PDG mapping, and row selection."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path

from .model import MAX_SUPPORTED_QUARK_LINES, LegacyOracleError, ProcessEntry

_PDG_BY_NAME = {
    "g": 21,
    "d": 1,
    "u": 2,
    "s": 3,
    "c": 4,
    "b": 5,
    "t": 6,
    "d~": -1,
    "u~": -2,
    "s~": -3,
    "c~": -4,
    "b~": -5,
    "t~": -6,
    "a": 22,
    "z": 23,
    "w+": 24,
    "w-": -24,
    "e+": -11,
    "e-": 11,
    "mu+": -13,
    "mu-": 13,
    "ta+": -15,
    "ta-": 15,
    "ve": 12,
    "ve~": -12,
    "vm": 14,
    "vm~": -14,
    "vt": 16,
    "vt~": -16,
    "h": 25,
}

# These rows were inspected at the pinned revision after applying the managed
# patch series. PDG multiset matching alone is not authoritative: several
# generated processes contain multiple rows with the same external PDGs.
EXPECTED_FORTRAN_PROCESS_ROWS: Mapping[str, ProcessEntry] = {
    "d d~ > z": ProcessEntry(1, 1, (1, -1, 23), (2, 1, 3)),
    "d d~ > z g": ProcessEntry(1, 1, (1, -1, 21, 23), (2, 3, 1, 4)),
    "d d~ > z g g": ProcessEntry(
        1,
        1,
        (1, -1, 21, 21, 23),
        (2, 3, 4, 1, 5),
    ),
}


def parse_process_file(path: Path) -> tuple[ProcessEntry, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise LegacyOracleError(f"empty process file: {path}")
    try:
        n_external, n_unique = (int(value) for value in lines[0].split())
    except (ValueError, TypeError) as error:
        raise LegacyOracleError(f"invalid process header in {path}") from error
    cursor = 1 + n_unique

    def next_nonempty() -> str:
        nonlocal cursor
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise LegacyOracleError(f"truncated process file: {path}")
        value = lines[cursor].strip()
        cursor += 1
        return value

    try:
        n_groups = int(next_nonempty())
        entries: list[ProcessEntry] = []
        for _ in range(n_groups):
            header = [int(value) for value in next_nonempty().split()]
            if len(header) != 3 + n_external:
                raise ValueError("invalid group header width")
            group, n_integrals, _max_channels = header[:3]
            for integral in range(1, n_integrals + 1):
                tokens = next_nonempty().split()
                n_channels = int(tokens[0])
                process_start = 1 + n_channels
                process_end = process_start + n_external
                order_end = process_end + n_external
                if len(tokens) != order_end + 1:
                    raise ValueError("invalid process row width")
                entries.append(
                    ProcessEntry(
                        group=group,
                        integral=integral,
                        process_pdgs=tuple(
                            int(value) for value in tokens[process_start:process_end]
                        ),
                        color_order=tuple(
                            int(value) for value in tokens[process_end:order_end]
                        ),
                    )
                )
    except ValueError as error:
        raise LegacyOracleError(f"invalid process file {path}: {error}") from error
    return tuple(entries)


def _normalized_process_expression(process: str) -> str:
    return " ".join(process.lower().replace("bar", "~").split())


def process_pdgs(process: str) -> tuple[int, ...]:
    parts = _normalized_process_expression(process).split(">")
    if len(parts) != 2:
        raise LegacyOracleError(f"invalid concrete process: {process!r}")
    names = (*parts[0].split(), *parts[1].split())
    try:
        return tuple(_PDG_BY_NAME[name] for name in names)
    except KeyError as error:
        raise LegacyOracleError(
            f"legacy built-in-SM oracle does not recognize {error.args[0]!r}"
        ) from error


def _validate_supported_quark_line_scope(
    pdgs: Sequence[int],
    *,
    context: str,
) -> int:
    quark_legs = sum(1 for pdg in pdgs if 1 <= abs(int(pdg)) <= 6)
    if quark_legs % 2:
        raise LegacyOracleError(
            f"{context}: legacy Fortran oracle cannot identify complete quark lines "
            f"from {quark_legs} external quark legs"
        )
    quark_lines = quark_legs // 2
    if quark_lines > MAX_SUPPORTED_QUARK_LINES:
        raise LegacyOracleError(
            f"{context}: {quark_lines} quark lines exceed the legacy Fortran "
            f"oracle scope of {MAX_SUPPORTED_QUARK_LINES}"
        )
    return quark_lines


def expected_process_entry(process: str) -> ProcessEntry:
    normalized = _normalized_process_expression(process)
    try:
        return EXPECTED_FORTRAN_PROCESS_ROWS[normalized]
    except KeyError as error:
        raise LegacyOracleError(
            f"no expected Fortran process row is declared for {process!r}"
        ) from error


def select_process_entry(entries: Sequence[ProcessEntry], process: str) -> ProcessEntry:
    selected, _matches = _select_declared_process_entry(
        entries,
        generated_process=process,
        wanted_pdgs=process_pdgs(process),
    )
    return selected


def _select_declared_process_entry(
    entries: Sequence[ProcessEntry],
    *,
    generated_process: str,
    wanted_pdgs: Sequence[int],
) -> tuple[ProcessEntry, tuple[ProcessEntry, ...]]:
    _validate_supported_quark_line_scope(
        wanted_pdgs,
        context=f"process {generated_process!r}",
    )
    expected = expected_process_entry(generated_process)
    if sorted(expected.process_pdgs) != sorted(int(pdg) for pdg in wanted_pdgs):
        raise LegacyOracleError(
            f"declared Fortran process row for {generated_process!r} has PDGs "
            f"{expected.process_pdgs}, incompatible with {tuple(wanted_pdgs)}"
        )
    matches = matching_process_entries_for_pdgs(entries, wanted_pdgs)
    if not matches:
        raise LegacyOracleError(f"no Fortran process row matches {generated_process!r}")
    declared_matches = tuple(entry for entry in matches if entry == expected)
    if len(declared_matches) > 1:
        raise LegacyOracleError(
            f"ambiguous duplicate expected Fortran rows for {generated_process!r}"
        )
    if not declared_matches:
        if len(matches) > 1:
            raise LegacyOracleError(
                f"ambiguous PDG-only Fortran rows for {generated_process!r}; "
                f"none matches declared row {expected}"
            )
        raise LegacyOracleError(
            f"Fortran row for {generated_process!r} is {matches[0]}, expected "
            f"declared row {expected}"
        )
    return declared_matches[0], matches


def matching_process_entries(
    entries: Sequence[ProcessEntry], process: str
) -> tuple[ProcessEntry, ...]:
    """Prefer the exact requested external order before multiset fallbacks."""

    return matching_process_entries_for_pdgs(entries, process_pdgs(process))


def matching_process_entries_for_pdgs(
    entries: Sequence[ProcessEntry],
    wanted: Sequence[int],
) -> tuple[ProcessEntry, ...]:
    """Match rows using an authoritative ordered external PDG sequence."""

    wanted = tuple(int(value) for value in wanted)
    wanted_multiset = sorted(wanted)
    matches = tuple(
        entry for entry in entries if sorted(entry.process_pdgs) == wanted_multiset
    )
    return tuple(entry for entry in matches if entry.process_pdgs == wanted) + tuple(
        entry for entry in matches if entry.process_pdgs != wanted
    )


def _permutation(
    source_pdgs: Sequence[int], target_pdgs: Sequence[int]
) -> tuple[int, ...]:
    positions: dict[int, deque[int]] = defaultdict(deque)
    for index, pdg in enumerate(source_pdgs):
        positions[int(pdg)].append(index)
    result: list[int] = []
    for pdg in target_pdgs:
        queue = positions[int(pdg)]
        if not queue:
            raise LegacyOracleError(
                "cannot map external ordering "
                f"{tuple(source_pdgs)} to {tuple(target_pdgs)}"
            )
        result.append(queue.popleft())
    if any(positions.values()):
        raise LegacyOracleError(
            f"cannot map external ordering {tuple(source_pdgs)} to {tuple(target_pdgs)}"
        )
    return tuple(result)


def _concrete_process_id(
    processes: Mapping[str, Mapping[str, object]],
    process_id: str,
) -> str:
    seen: set[str] = set()
    current_id = process_id
    while processes[current_id]["alias_of"] is not None:
        if current_id in seen:
            raise LegacyOracleError(f"process alias cycle includes {current_id}")
        seen.add(current_id)
        current_id = str(processes[current_id]["alias_of"])
        if current_id not in processes:
            raise LegacyOracleError(
                f"process {process_id} aliases unknown process {current_id}"
            )
    return current_id


def _ordered_leg_ids(
    leg_ids: Sequence[str],
    positions: Sequence[int],
    *,
    context: str,
) -> tuple[str, ...]:
    try:
        result = tuple(leg_ids[position - 1] for position in positions)
    except IndexError as error:
        raise LegacyOracleError(
            f"{context} references an external position outside 1..{len(leg_ids)}"
        ) from error
    if any(position < 1 for position in positions):
        raise LegacyOracleError(
            f"{context} references an external position outside 1..{len(leg_ids)}"
        )
    return result
