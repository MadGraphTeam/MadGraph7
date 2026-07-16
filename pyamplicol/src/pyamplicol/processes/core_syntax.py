# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass

ParticleName = str
ProcessTuple = tuple[ParticleName, ...]
OrderTuple = tuple[int, ...]


QUARKS = frozenset({"d", "u", "s", "c", "b", "t"})
ANTIQUARKS = frozenset({f"{q}~" for q in QUARKS})
SINGLETS = frozenset(
    {
        "a",
        "z",
        "w+",
        "w-",
        "e+",
        "e-",
        "mu+",
        "mu-",
        "ta+",
        "ta-",
        "ve",
        "ve~",
        "vm",
        "vm~",
        "vt",
        "vt~",
        "h",
    }
)
GLUONS = frozenset({"g"})
ALL_COLOURED = QUARKS | ANTIQUARKS | GLUONS
PDGS = {
    "g": "21",
    "d": "1",
    "u": "2",
    "s": "3",
    "c": "4",
    "b": "5",
    "t": "6",
    "d~": "-1",
    "u~": "-2",
    "s~": "-3",
    "c~": "-4",
    "b~": "-5",
    "t~": "-6",
    "a": "22",
    "z": "23",
    "w+": "24",
    "w-": "-24",
    "e+": "-11",
    "e-": "11",
    "mu+": "-13",
    "mu-": "13",
    "ta+": "-15",
    "ta-": "15",
    "ve": "12",
    "ve~": "-12",
    "vm": "14",
    "vm~": "-14",
    "vt": "16",
    "vt~": "-16",
    "h": "25",
}
ANTI_PARTICLE = {
    "g": "g",
    "d": "d~",
    "u": "u~",
    "s": "s~",
    "c": "c~",
    "b": "b~",
    "t": "t~",
    "d~": "d",
    "u~": "u",
    "s~": "s",
    "c~": "c",
    "b~": "b",
    "t~": "t",
    "a": "a",
    "z": "z",
    "w+": "w-",
    "w-": "w+",
    "e+": "e-",
    "e-": "e+",
    "mu+": "mu-",
    "mu-": "mu+",
    "ta+": "ta-",
    "ta-": "ta+",
    "ve": "ve~",
    "ve~": "ve",
    "vm": "vm~",
    "vm~": "vm",
    "vt": "vt~",
    "vt~": "vt",
    "h": "h",
}
SORT_PARTICLES = {
    "g": 13,
    "d": 1,
    "u": 2,
    "s": 3,
    "c": 4,
    "b": 5,
    "t": 6,
    "d~": 7,
    "u~": 8,
    "s~": 9,
    "c~": 10,
    "b~": 11,
    "t~": 12,
    "a": 80,
    "z": 81,
    "w+": 82,
    "w-": 83,
    "e+": 84,
    "e-": 85,
    "mu+": 86,
    "mu-": 87,
    "ta+": 88,
    "ta-": 89,
    "ve": 90,
    "ve~": 91,
    "vm": 92,
    "vm~": 93,
    "vt": 94,
    "vt~": 95,
    "h": 96,
}
CHARGES3 = {
    "g": 0,
    "d": -1,
    "u": 2,
    "s": -1,
    "c": 2,
    "b": -1,
    "t": 2,
    "d~": 1,
    "u~": -2,
    "s~": 1,
    "c~": -2,
    "b~": 1,
    "t~": -2,
    "a": 0,
    "z": 0,
    "w+": 3,
    "w-": -3,
    "e+": 3,
    "e-": -3,
    "mu+": 3,
    "mu-": -3,
    "ta+": 3,
    "ta-": -3,
    "ve": 0,
    "ve~": 0,
    "vm": 0,
    "vm~": 0,
    "vt": 0,
    "vt~": 0,
    "h": 0,
}
FAMILY = {
    "g": 0,
    "d": 1,
    "u": 1,
    "s": 11,
    "c": 11,
    "b": 21,
    "t": 21,
    "d~": -1,
    "u~": -1,
    "s~": -11,
    "c~": -11,
    "b~": -21,
    "t~": -21,
    "a": 0,
    "z": 0,
    "w+": 0,
    "w-": 0,
    "e+": -31,
    "e-": 31,
    "mu+": -41,
    "mu-": 41,
    "ta+": -51,
    "ta-": 51,
    "ve": 31,
    "ve~": -31,
    "vm": 41,
    "vm~": -41,
    "vt": 51,
    "vt~": -51,
    "h": 0,
}


@dataclass(frozen=True)
class ParticleSelectionMetadata:
    name: str
    pdg: int
    anti_name: str
    charge3: int
    family: int
    color_rep: int
    is_fermion: bool
    is_antiparticle: bool

    @property
    def is_colored(self) -> bool:
        return self.color_rep != 1


def _selection_metadata(name: str) -> ParticleSelectionMetadata:
    return _PARTICLE_SELECTION_METADATA[name]


def _metadata_color_rep(name: str) -> int:
    if name in QUARKS:
        return 3
    if name in ANTIQUARKS:
        return -3
    if name in GLUONS:
        return 8
    return 1


_PARTICLE_SELECTION_METADATA = {
    name: ParticleSelectionMetadata(
        name=name,
        pdg=int(PDGS[name]),
        anti_name=ANTI_PARTICLE[name],
        charge3=CHARGES3[name],
        family=FAMILY[name],
        color_rep=_metadata_color_rep(name),
        is_fermion=name in QUARKS
        or name in ANTIQUARKS
        or abs(int(PDGS[name])) in range(11, 17),
        is_antiparticle=name in ANTIQUARKS or int(PDGS[name]) < 0,
    )
    for name in PDGS
}


@dataclass(frozen=True)
class ProcessOptions:
    flavour_scheme: int = 5
    include_3qqbar: bool = False
    include_cc: bool = False
    include_resonance: bool = False
    serial: bool = True


@dataclass(frozen=True)
class ParsedProcess:
    initial_state: ProcessTuple
    jet_count: int
    rest: ProcessTuple
    leptons: ProcessTuple = ()


@dataclass(frozen=True)
class SubprocessRecord:
    process: ProcessTuple
    color_order: OrderTuple
    multichannel_partners: tuple[int, ...] = ()
    identical_factor: float = 1.0


@dataclass(frozen=True)
class PhaseSpaceGroup:
    group_id: int
    phase_space_order: OrderTuple
    records: tuple[SubprocessRecord, ...]


@dataclass(frozen=True)
class ProcessEnumeration:
    request: ParsedProcess
    options: ProcessOptions
    unique_processes: tuple[ProcessTuple, ...]
    groups: tuple[PhaseSpaceGroup, ...]

    @property
    def n_external(self) -> int:
        if self.unique_processes:
            return len(self.unique_processes[0])
        if self.groups and self.groups[0].records:
            return len(self.groups[0].records[0].process)
        return 0

    @property
    def n_records(self) -> int:
        return sum(len(group.records) for group in self.groups)


@dataclass(frozen=True)
class ProcessSetEntry:
    key: str
    process: str
    enumeration: ProcessEnumeration


@dataclass(frozen=True)
class ProcessSelectionRecord:
    source: str
    process: str
    key: str
    status: str
    reason: str
    quark_lines: int
    charge3: int
    family: int


@dataclass(frozen=True)
class ProcessSelectionReport:
    request: str
    options: ProcessOptions
    entries: tuple[ProcessSetEntry, ...]
    records: tuple[ProcessSelectionRecord, ...]
    candidate_count: int
    evaluated_count: int
    selected_count: int
    duplicate_count: int
    rejected_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    stage_timings: tuple[tuple[str, float], ...]
    elapsed_s: float
    prefilter_enabled: bool

    @property
    def selected_records(self) -> tuple[ProcessSelectionRecord, ...]:
        return tuple(record for record in self.records if record.status == "selected")

    @property
    def duplicate_records(self) -> tuple[ProcessSelectionRecord, ...]:
        return tuple(record for record in self.records if record.status == "duplicate")


@dataclass(frozen=True)
class ProcessSetEnumeration:
    request: str
    options: ProcessOptions
    entries: tuple[ProcessSetEntry, ...]
    selection_report: ProcessSelectionReport | None = None

    @property
    def default_key(self) -> str:
        if not self.entries:
            raise ValueError("process set is empty")
        return self.entries[0].key


def split_process_set(process_string: str) -> tuple[str, ...]:
    """Split process sets without treating bars inside brackets as separators."""

    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(process_string):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unmatched ']' in process string")
        elif char == "|" and depth == 0:
            part = process_string[start:index].strip()
            if not part:
                raise ValueError("empty process in process set")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise ValueError("unmatched '[' in process string")
    tail = process_string[start:].strip()
    if not tail:
        raise ValueError("empty process in process set")
    parts.append(tail)
    return tuple(parts)


def expand_process_variants(process_string: str) -> tuple[str, ...]:
    """Expand anonymous multiparticle slots and repetition syntax.

    Built-in inclusive labels such as ``p`` and ``j`` are kept symbolic for the
    enumerator. Anonymous slots like ``[d g]`` are expanded by cartesian product,
    and each repeated slot in ``3*[d g]`` is treated independently.
    """

    variants: list[str] = []
    for process in split_process_set(process_string):
        parts = process.lower().replace("bar", "~").split(">")
        if len(parts) != 2:
            raise ValueError("invalid collision format; expected 'initial > final'")
        initial_options = _expand_side_tokens(_tokenize_side(parts[0].strip()))
        final_options = _expand_side_tokens(_tokenize_side(parts[1].strip()))
        for initial in itertools.product(*initial_options):
            for final in itertools.product(*final_options):
                variants.append(f"{' '.join(initial)} > {' '.join(final)}")
    return tuple(dict.fromkeys(variants))


def canonical_process_key(process: str) -> str:
    tokens = process.lower().replace("bar", "~").replace(">", " > ").split()
    safe = []
    for token in tokens:
        if token == ">":
            safe.append("to")
        else:
            safe.append(
                token.replace("~", "bar").replace("+", "plus").replace("-", "minus")
            )
    return "_".join(safe)


def _process_uses_inclusive_labels(process: str) -> bool:
    parts = process.lower().replace("bar", "~").split(">")
    if len(parts) != 2:
        raise ValueError("invalid collision format; expected 'initial > final'")
    for token in (*_tokenize_side(parts[0].strip()), *_tokenize_side(parts[1].strip())):
        _, item = _split_repeat_token(token)
        if item in {"p", "j"}:
            return True
        if (
            item.startswith("[")
            and item.endswith("]")
            and any(option in {"p", "j"} for option in _anonymous_options(item))
        ):
            return True
    return False


def _request_allows_charged_current(request: ParsedProcess) -> bool:
    """Return whether request quantum numbers imply charged-current flow."""

    if "w+" in request.rest or "w-" in request.rest:
        return True
    non_qcd_rest = [
        particle
        for particle in request.rest
        if particle not in ALL_COLOURED and particle in CHARGES3
    ]
    return abs(sum(CHARGES3[particle] for particle in non_qcd_rest)) == 3


def _record_to_physical_process(record: SubprocessRecord) -> str:
    initial = tuple(ANTI_PARTICLE[p] for p in record.process[:2])
    final = record.process[2:]
    return f"{' '.join(initial)} > {' '.join(final)}"


def _concrete_processes_from_inclusive_enumeration(
    enumeration: ProcessEnumeration,
) -> tuple[str, ...]:
    processes: dict[str, None] = {}
    for group in enumeration.groups:
        for record in group.records:
            processes.setdefault(_record_to_physical_process(record), None)
    return tuple(sorted(processes, key=_concrete_process_sort_key))


def _concrete_process_sort_key(process: str) -> tuple[object, ...]:
    initial, _, final = process.partition(">")
    initial_tokens = tuple(_tokenize_side(initial.strip()))
    final_tokens = tuple(_tokenize_side(final.strip()))
    q_qbar_first = not (
        len(initial_tokens) == 2
        and not initial_tokens[0].endswith("~")
        and initial_tokens[1].endswith("~")
    )
    return (
        q_qbar_first,
        tuple(SORT_PARTICLES.get(token, 999) for token in initial_tokens),
        tuple(SORT_PARTICLES.get(token, 999) for token in final_tokens),
        initial_tokens,
        final_tokens,
    )


def _tokenize_side(side: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(side):
        if side[index].isspace():
            index += 1
            continue
        if side[index] == "[":
            end = side.find("]", index + 1)
            if end < 0:
                raise ValueError("unmatched '[' in process string")
            tokens.append(side[index : end + 1])
            index = end + 1
            continue
        end = index
        while end < len(side) and not side[end].isspace():
            if side[end] == "[":
                bracket = side.find("]", end + 1)
                if bracket < 0:
                    raise ValueError("unmatched '[' in process string")
                end = bracket + 1
            else:
                end += 1
        tokens.append(side[index:end])
        index = end
    return tokens


def _expand_side_tokens(tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    expanded: list[tuple[str, ...]] = []
    for token in tokens:
        repeat, item = _split_repeat_token(token)
        options = _anonymous_options(item)
        expanded.extend(options for _ in range(repeat))
    return tuple(expanded)


def _split_repeat_token(token: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)\*(.+)", token)
    if match:
        return int(match.group(1)), match.group(2)
    compact = re.fullmatch(r"(\d+)([A-Za-z][A-Za-z0-9+~\-]*)", token)
    if compact:
        return int(compact.group(1)), compact.group(2)
    return 1, token


def _anonymous_options(token: str) -> tuple[str, ...]:
    if token.startswith("[") and token.endswith("]"):
        options = tuple(_tokenize_side(token[1:-1].strip()))
        if not options:
            raise ValueError("anonymous multiparticle label cannot be empty")
        return options
    return (token,)


def _ordered_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts <= 0:
        return ((),) if total == 0 else ()
    if parts == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total + 1)
        for rest in _ordered_compositions(total - first, parts - 1)
    )


def _chunk_sequence(
    values: Sequence[int],
    chunk_lengths: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    chunks: list[tuple[int, ...]] = []
    start = 0
    for length in chunk_lengths:
        chunks.append(tuple(values[start : start + length]))
        start += length
    return tuple(chunks)
