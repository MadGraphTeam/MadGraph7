# SPDX-License-Identifier: 0BSD
"""The data file ``reweight_<id>.rwd`` of a reweight provider.

The generated C header only names the file and its SHA-256; ``reweight_data.h``
maps and validates it at ``umami_initialize``. Layout (little-endian, sections at
8-byte aligned offsets): a 128-byte header, a section table of 32-byte entries
``{u32 tag, elem_size; u64 offset, count, reserved}``, then the sections. A ref
is an element offset into the int32 pool ``I32P``, a str a byte offset into
``STRS``; ``0xFFFFFFFF`` is NULL for both. The record layouts are the
``RWD_*`` structs of ``reweight_data.h``.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import zlib
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderScalars:
    id: str
    particle_count: int
    exported_helicity_count: int
    reference_artifact: str
    target_artifact: str
    reference_artifact_id: str | None
    target_artifact_id: str | None
    supports_alpha_s: bool


@dataclass(frozen=True)
class ProcessRow:
    key: str
    pdgs: tuple[int, ...]
    alpha_s_name: str | None
    # -1 if unproven
    alpha_s_power: int
    helicity_count: int


@dataclass(frozen=True)
class FlavourRow:
    """One ReweightFlavour; target_permutation None: the reference permutation."""

    reference_process: int
    target_process: int
    permutation: tuple[int, ...]
    reference_signs: tuple[int, ...] | None
    reference_helicity: tuple[int, ...]
    target_helicity: tuple[int, ...]
    reference_assignment: int
    assignment_key: str | None
    target_permutation: tuple[int, ...] | None
    target_signs: tuple[int, ...] | None
    target_factor: float
    target_assignment: int
    target_assignment_key: str | None
    switch_arrangement: int


@dataclass(frozen=True)
class SwitchEvaluation:
    process: int
    legs: tuple[int, ...]
    assignment: int
    assignment_key: str | None
    signs: tuple[int, ...] | None
    helicity_signs: tuple[int, ...] | None


@dataclass(frozen=True)
class SwitchCall:
    """Evaluations of one native call; assignments None: a plain evaluation (one
    entry, no entries or permutations)."""

    evaluations: tuple[int, ...]
    assignments: tuple[int, ...] | None
    entries: tuple[int, ...] | None
    # the relabelling of each request, particle_count values each (None: identity)
    permutations: tuple[int, ...] | None


@dataclass(frozen=True)
class SwitchOrderings:
    """The orderings by column (millions of rows at 7j)."""

    evaluation: array
    pdf0: array
    pdf1: array
    weight: array
    target: array
    arrangement: array


@dataclass(frozen=True)
class SwitchTargets:
    """The targets by column; legs None: the evaluation's legs."""

    evaluation: array
    target_process: array
    colors: list[tuple[int, ...]]
    flows: list[tuple[int, ...]]
    legs: list[tuple[int, ...] | None]
    factor: array
    assignment: array
    assignment_key: list[str | None]


@dataclass(frozen=True)
class SwitchRows:
    evaluations: list[SwitchEvaluation]
    calls: list[SwitchCall]
    orderings: SwitchOrderings
    targets: SwitchTargets
    # flow ids per reference process
    flows: list[tuple[str, ...]]
    # the exported helicities, particle_count values each
    helicities: tuple[int, ...]
    arrangement_count: int


@dataclass(frozen=True)
class ReweightTables:
    """The data of one reweight provider, see ``reweight_provider.h``."""

    provider: ProviderScalars
    reference_processes: list[ProcessRow]
    target_processes: list[ProcessRow]
    channels: list[list[FlavourRow]]
    switch: SwitchRows | None


MAGIC = b"PYACRWD\0"
FORMAT_VERSION = 1
HEADER_SIZE = 128
ENDIAN_TAG = 0x01020304
NULL = 0xFFFFFFFF
FLAG_ALPHA_S = 1
FLAG_SWITCH = 2
# parton density columns: PDG id + 6
PDF_COLUMNS = 13

_HEADER = struct.Struct("<8sIIIIQII32s56s")
_ENTRY = struct.Struct("<IIQQQ")
# tag -> record struct (its size is the section's elem_size)
_RECORDS = {
    b"PROV": struct.Struct("<8I6I"),
    b"STRS": struct.Struct("<B"),
    b"I32P": struct.Struct("<i"),
    b"F64T": struct.Struct("<d"),
    b"PROC": struct.Struct("<IIIiI"),
    b"CHAN": struct.Struct("<II"),
    b"FLAV": struct.Struct("<II6IiIiIIi"),
    b"SEVL": struct.Struct("<IIiIII"),
    b"SCAL": struct.Struct("<IIIIII"),
    b"SORD": struct.Struct("<IiIHBB"),
    b"STGT": struct.Struct("<IIIIIiII"),
    b"SFLW": struct.Struct("<II"),
    b"SFID": struct.Struct("<I"),
    b"SHEL": struct.Struct("<i"),
}
_ORDER = tuple(_RECORDS)


def _producer() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # the source tree without a build
        return "unknown"


class _Pools:
    """The interned int32 arrays, strings and doubles, in order of first use."""

    def __init__(self) -> None:
        self.ints = array("i")
        if self.ints.itemsize != 4:
            raise RuntimeError("the reweight data needs a 4-byte array('i')")
        self._arrays: dict[tuple, int] = {}
        # by identity first: the tables share equal arrays as one tuple (kept
        # alive here, so that no id is reused)
        self._by_id: dict[int, tuple[int, tuple]] = {}
        self.strings = bytearray()
        self._strings: dict[str, int] = {}
        self.doubles = array("d")
        self._doubles: dict[int, int] = {}

    def ref(self, values: Sequence[int] | None) -> int:
        if values is None:
            return NULL
        known = self._by_id.get(id(values))
        if known is not None:
            return known[0]
        key = tuple(values)
        offset = self._arrays.get(key)
        if offset is None:
            # an empty array is any offset with length 0
            offset = len(self.ints) if key else 0
            if key:
                self.ints.extend(key)
            if len(self.ints) >= NULL:
                raise ValueError("the reweight data int pool is too large")
            self._arrays[key] = offset
        if isinstance(values, tuple):
            self._by_id[id(values)] = (offset, values)
        return offset

    def text(self, value: str | None) -> int:
        if value is None:
            return NULL
        offset = self._strings.get(value)
        if offset is None:
            if "\0" in value:
                raise ValueError(f"reweight data strings cannot contain NUL: {value!r}")
            offset = len(self.strings)
            self.strings += value.encode("utf-8") + b"\0"
            if len(self.strings) >= NULL:
                raise ValueError("the reweight data string pool is too large")
            self._strings[value] = offset
        return offset

    def double(self, value: float) -> int:
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        index = self._doubles.get(bits)
        if index is None:
            index = self._doubles[bits] = len(self.doubles)
            self.doubles.append(value)
        return index

    def doubles_of(self, values: array) -> array:
        """The constant index of every value of an array('d')."""
        bits = memoryview(values).cast("B").cast("Q")
        known = self._doubles
        result = array("I", bytes(4 * len(values)))
        for k, pattern in enumerate(bits):
            index = known.get(pattern)
            if index is None:
                index = known[pattern] = len(self.doubles)
                self.doubles.append(values[k])
            result[k] = index
        return result


def _u32(values) -> array:
    # two's complement of signed columns
    return array("I", (v & 0xFFFFFFFF for v in values))


def _interleave(columns: Sequence[array], count: int) -> array:
    width = len(columns)
    out = array("I", bytes(4 * width * count))
    for k, column in enumerate(columns):
        out[k::width] = column
    return _little(out)


def _little(values: array) -> array:
    # an array is written as it is (no copy) on little-endian hosts
    if sys.byteorder != "little":
        values = array(values.typecode, values)
        values.byteswap()
    return values


def _range(name: str, values, low: int, high: int) -> None:
    for value in values:
        if not low <= value < high:
            raise ValueError(f"reweight data {name} {value} is outside [{low}, {high})")


def encode_reweight_data(tables: ReweightTables) -> bytes:
    """The data file of the tables; deterministic (interning in order of first use)."""
    header, chunks = _encode(tables)
    return header + b"".join(chunks)


def _encode(tables: ReweightTables) -> tuple[bytes, list]:
    """The header and the payload chunks of the data file (written without
    joining them: the payload is most of the writer's memory at 7j)."""
    pools = _Pools()
    p = tables.provider
    n = p.particle_count
    sections: dict[bytes, bytes] = {}

    def records(tag: bytes, rows) -> bytes:
        pack = _RECORDS[tag].pack
        return b"".join(pack(*row) for row in rows)

    switch = tables.switch
    flags = (FLAG_ALPHA_S if p.supports_alpha_s else 0) | (FLAG_SWITCH if switch else 0)
    provider = (
        n,
        p.exported_helicity_count,
        len(tables.reference_processes),
        len(tables.target_processes),
        len(tables.channels),
        flags,
        0 if switch is None else switch.arrangement_count,
        0,
        pools.text(p.id),
        pools.text(p.reference_artifact),
        pools.text(p.target_artifact),
        pools.text(p.reference_artifact_id),
        pools.text(p.target_artifact_id),
        pools.text(_producer()),
    )
    sections[b"PROV"] = _RECORDS[b"PROV"].pack(*provider)
    sections[b"PROC"] = records(
        b"PROC",
        (
            (
                pools.text(q.key),
                pools.ref(q.pdgs),
                pools.text(q.alpha_s_name),
                q.alpha_s_power,
                q.helicity_count,
            )
            for q in (*tables.reference_processes, *tables.target_processes)
        ),
    )
    channels, flavours, first = [], [], 0
    for rows in tables.channels:
        channels.append((first, len(rows)))
        first += len(rows)
        flavours.extend(rows)
    sections[b"CHAN"] = records(b"CHAN", channels)
    sections[b"FLAV"] = records(
        b"FLAV",
        (
            (
                f.reference_process,
                f.target_process,
                pools.ref(f.permutation),
                pools.ref(f.reference_signs),
                pools.ref(f.reference_helicity),
                pools.ref(f.target_helicity),
                pools.ref(f.target_permutation),
                pools.ref(f.target_signs),
                f.reference_assignment,
                pools.text(f.assignment_key),
                f.target_assignment,
                pools.text(f.target_assignment_key),
                pools.double(f.target_factor),
                f.switch_arrangement,
            )
            for f in flavours
        ),
    )
    if switch is None:
        for tag in (b"SEVL", b"SCAL", b"SORD", b"STGT", b"SFLW", b"SFID", b"SHEL"):
            sections[tag] = b""
    else:
        sections[b"SEVL"] = records(
            b"SEVL",
            (
                (
                    e.process,
                    pools.ref(e.legs),
                    e.assignment,
                    pools.text(e.assignment_key),
                    pools.ref(e.signs),
                    pools.ref(e.helicity_signs),
                )
                for e in switch.evaluations
            ),
        )
        sections[b"SCAL"] = records(
            b"SCAL",
            (
                (
                    len(c.evaluations),
                    pools.ref(c.evaluations),
                    pools.ref(c.assignments),
                    1 if c.assignments is None else len(c.assignments),
                    pools.ref(c.permutations),
                    pools.ref(c.entries),
                )
                for c in switch.calls
            ),
        )
        o = switch.orderings
        count = len(o.evaluation)
        _range("ordering arrangement", o.arrangement, 0, 1 << 16)
        _range("ordering parton density", o.pdf0, 0, PDF_COLUMNS)
        _range("ordering parton density", o.pdf1, 0, PDF_COLUMNS)
        packed = array(
            "I",
            (
                a | b << 16 | c << 24
                for a, b, c in zip(o.arrangement, o.pdf0, o.pdf1, strict=True)
            ),
        )
        sections[b"SORD"] = _interleave(
            (
                array("I", o.evaluation),
                _u32(o.target),
                pools.doubles_of(o.weight),
                packed,
            ),
            count,
        )
        t = switch.targets
        sections[b"STGT"] = _interleave(
            (
                array("I", t.evaluation),
                array("I", t.target_process),
                array("I", map(pools.ref, t.colors)),
                array("I", map(pools.ref, t.flows)),
                array("I", map(pools.ref, t.legs)),
                _u32(t.assignment),
                pools.doubles_of(t.factor),
                array("I", map(pools.text, t.assignment_key)),
            ),
            len(t.evaluation),
        )
        ids, rows = [], []
        for flow_ids in switch.flows:
            rows.append((len(ids), len(flow_ids)))
            ids.extend(pools.text(i) for i in flow_ids)
        sections[b"SFLW"] = records(b"SFLW", rows)
        sections[b"SFID"] = _little(array("I", ids))
        sections[b"SHEL"] = _little(array("i", switch.helicities))
    if not pools.strings:
        pools.strings += b"\0"
    sections[b"STRS"] = bytes(pools.strings)
    sections[b"I32P"] = _little(pools.ints)
    sections[b"F64T"] = _little(pools.doubles)
    position = HEADER_SIZE + _ENTRY.size * len(_ORDER)
    entries, data_chunks = [], []
    for tag in _ORDER:
        data = memoryview(sections[tag]).cast("B")
        size = _RECORDS[tag].size
        if len(data) % size:
            raise AssertionError(f"section {tag!r} is not a multiple of {size}")
        pad = -position % 8
        data_chunks.append(b"\0" * pad)
        position += pad
        entries.append(
            _ENTRY.pack(int.from_bytes(tag, "little"), size, position, len(data) // size, 0)
        )
        data_chunks.append(data)
        position += len(data)
    chunks = [b"".join(entries), *data_chunks]
    digest, crc = hashlib.sha256(), 0
    for chunk in chunks:
        digest.update(chunk)
        crc = zlib.crc32(chunk, crc)
    header = bytearray(
        _HEADER.pack(
            MAGIC,
            FORMAT_VERSION,
            HEADER_SIZE,
            ENDIAN_TAG,
            len(_ORDER),
            position,
            crc,
            0,
            digest.digest(),
            b"\0" * 56,
        )
    )
    struct.pack_into("<I", header, 36, zlib.crc32(header))
    return bytes(header), chunks


def reseal_reweight_data(data: bytes) -> bytes:
    """The file with its checksums and digest recomputed (tests of the loader)."""
    body = data[HEADER_SIZE:]
    header = bytearray(data[:HEADER_SIZE])
    struct.pack_into("<I", header, 32, zlib.crc32(body))
    header[40:72] = hashlib.sha256(body).digest()
    struct.pack_into("<I", header, 36, 0)
    struct.pack_into("<I", header, 36, zlib.crc32(header))
    return bytes(header) + body


def section_table(data: bytes) -> dict[str, tuple[int, int, int]]:
    """Each section's (offset, count, element size), unchecked."""
    count = struct.unpack_from("<I", data, 20)[0]
    table = {}
    for k in range(count):
        tag, size, offset, rows, _ = _ENTRY.unpack_from(
            data, HEADER_SIZE + _ENTRY.size * k
        )
        table[tag.to_bytes(4, "little").decode()] = (offset, rows, size)
    return table


def content_sha256(data: bytes) -> str:
    """The digest bound into the library (REWEIGHT_DATA_SHA256)."""
    return data[40:72].hex()


class _Reader:
    def __init__(self, data: bytes) -> None:
        if len(data) < HEADER_SIZE:
            raise ValueError("reweight data: file is truncated")
        (
            magic,
            version,
            header_size,
            endian,
            section_count,
            file_size,
            payload_crc,
            header_crc,
            digest,
            _reserved,
        ) = _HEADER.unpack_from(data)
        if magic != MAGIC:
            raise ValueError("reweight data: not a reweight data file")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"reweight data has format version {version}, this reads "
                f"{FORMAT_VERSION}: regenerate the output"
            )
        if endian != ENDIAN_TAG or header_size != HEADER_SIZE:
            raise ValueError("reweight data: wrong byte order or header size")
        if file_size != len(data):
            raise ValueError("reweight data: file size mismatch (truncated?)")
        header = bytearray(data[:HEADER_SIZE])
        header[36:40] = b"\0\0\0\0"
        if zlib.crc32(header) != header_crc:
            raise ValueError("reweight data: header checksum mismatch")
        if HEADER_SIZE + _ENTRY.size * section_count > file_size:
            raise ValueError("reweight data: section table out of bounds")
        if zlib.crc32(memoryview(data)[HEADER_SIZE:]) != payload_crc:
            raise ValueError("reweight data: payload checksum mismatch (corrupted)")
        if hashlib.sha256(memoryview(data)[HEADER_SIZE:]).digest() != digest:
            raise ValueError("reweight data: content digest mismatch")
        self.data = data
        self.sections: dict[bytes, tuple[int, int]] = {}
        for k in range(section_count):
            tag, size, offset, count, _ = _ENTRY.unpack_from(
                data, HEADER_SIZE + _ENTRY.size * k
            )
            tag = tag.to_bytes(4, "little")
            record = _RECORDS.get(tag)
            if record is None or tag in self.sections:
                raise ValueError(f"reweight data: unknown or repeated section {tag!r}")
            if size != record.size or offset % 8 or offset + size * count > file_size:
                raise ValueError(f"reweight data: section {tag!r} is malformed")
            self.sections[tag] = (offset, count)
        if set(self.sections) != set(_RECORDS):
            raise ValueError("reweight data: sections are missing")

    def count(self, tag: bytes) -> int:
        return self.sections[tag][1]

    def rows(self, tag: bytes) -> list[tuple]:
        offset, count = self.sections[tag]
        record = _RECORDS[tag]
        return list(
            record.iter_unpack(self.data[offset : offset + record.size * count])
        )

    def column(self, tag: bytes, typecode: str) -> array:
        offset, count = self.sections[tag]
        values = array(typecode)
        values.frombytes(self.data[offset : offset + _RECORDS[tag].size * count])
        if sys.byteorder != "little":
            values.byteswap()
        return values


def decode_reweight_data(data: bytes) -> ReweightTables:
    """The tables of a data file, after the checks of the C loader."""
    r = _Reader(data)
    strings = bytes(r.column(b"STRS", "B"))
    if not strings.endswith(b"\0"):
        raise ValueError("reweight data: the string pool is not terminated")
    pool = r.column(b"I32P", "i")
    doubles = r.column(b"F64T", "d")

    def text(offset: int, required: bool = False) -> str | None:
        if offset == NULL:
            if required:
                raise ValueError("reweight data: a required string is NULL")
            return None
        if offset >= len(strings):
            raise ValueError("reweight data: string offset out of range")
        return strings[offset : strings.index(b"\0", offset)].decode("utf-8")

    def ints(ref: int, length: int, low=None, high=None, nullable=True):
        if ref == NULL:
            if not nullable:
                raise ValueError("reweight data: a required array is NULL")
            return None
        if ref + length > len(pool):
            raise ValueError("reweight data: array out of range")
        values = tuple(pool[ref : ref + length])
        if low is not None:
            _range("array value", values, low, high)
        return values

    def double(index: int) -> float:
        if index >= len(doubles):
            raise ValueError("reweight data: constant index out of range")
        return doubles[index]

    (prov,) = r.rows(b"PROV") if r.count(b"PROV") == 1 else [None]
    if prov is None:
        raise ValueError("reweight data: needs one provider record")
    (
        n,
        exported,
        n_ref,
        n_tgt,
        n_chan,
        flags,
        arrangements,
        _,
        s_id,
        s_ref,
        s_tgt,
        s_ref_id,
        s_tgt_id,
        _producer_ref,
    ) = prov
    if n_ref < 1 or n_tgt < 1:
        raise ValueError("reweight data: needs reference and target processes")
    provider = ProviderScalars(
        text(s_id, True),
        n,
        exported,
        text(s_ref, True),
        text(s_tgt, True),
        text(s_ref_id),
        text(s_tgt_id),
        bool(flags & FLAG_ALPHA_S),
    )
    if r.count(b"PROC") != n_ref + n_tgt:
        raise ValueError("reweight data: process count mismatch")
    processes = [
        ProcessRow(
            text(key, True), ints(pdgs, n, nullable=False), text(name), power, hel
        )
        for key, pdgs, name, power, hel in r.rows(b"PROC")
    ]
    flavours = r.rows(b"FLAV")
    channel_rows = r.rows(b"CHAN")
    if len(channel_rows) != n_chan:
        raise ValueError("reweight data: channel count mismatch")
    channels = []
    for first, count in channel_rows:
        if first + count > len(flavours):
            raise ValueError("reweight data: channel out of range")
        rows = []
        for (
            rp,
            tp,
            perm,
            rsigns,
            rhel,
            thel,
            tperm,
            tsigns,
            rass,
            key,
            tass,
            tkey,
            factor,
            arrangement,
        ) in flavours[first : first + count]:
            if rp >= n_ref or tp >= n_tgt:
                raise ValueError("reweight data: flavour process out of range")
            if (rass >= 0 and key == NULL) or (tass >= 0 and tkey == NULL):
                raise ValueError("reweight data: assignment without its process id")
            rows.append(
                FlavourRow(
                    rp,
                    tp,
                    ints(perm, n, 0, n, False),
                    ints(rsigns, n, -1, 2),
                    ints(rhel, exported, -1, processes[rp].helicity_count, False),
                    ints(
                        thel, exported, -1, processes[n_ref + tp].helicity_count, False
                    ),
                    rass,
                    text(key),
                    ints(tperm, n, 0, n),
                    ints(tsigns, n, -1, 2),
                    double(factor),
                    tass,
                    text(tkey),
                    arrangement,
                )
            )
            for signs in (rows[-1].reference_signs, rows[-1].target_signs):
                if signs is not None and 0 in signs:
                    raise ValueError("reweight data: signs must be -1 or 1")
        channels.append(rows)
    switch = None
    if flags & FLAG_SWITCH:
        evaluations = []
        for process, legs, assignment, key, signs, hsigns in r.rows(b"SEVL"):
            if process >= n_ref or (assignment >= 0 and key == NULL):
                raise ValueError("reweight data: switch evaluation out of range")
            evaluation = SwitchEvaluation(
                process,
                ints(legs, n, 0, n, False),
                assignment,
                text(key),
                ints(signs, n, -1, 2),
                ints(hsigns, n, -1, 2),
            )
            if 0 in (evaluation.signs or ()) or 0 in (evaluation.helicity_signs or ()):
                raise ValueError("reweight data: signs must be -1 or 1")
            evaluations.append(evaluation)
        n_eval = len(evaluations)
        calls = []
        for count, members, assignments, entry_count, permutations, entries in r.rows(
            b"SCAL"
        ):
            if count < 1:
                raise ValueError("reweight data: a switch call without evaluations")
            members = ints(members, count, 0, n_eval, False)
            if assignments == NULL:
                plain = count == 1 and entry_count == 1
                if not plain or permutations != NULL or entries != NULL:
                    raise ValueError("reweight data: malformed plain switch call")
                calls.append(SwitchCall(members, None, None, None))
                continue
            calls.append(
                SwitchCall(
                    members,
                    ints(assignments, entry_count),
                    ints(entries, count, 0, entry_count, False),
                    ints(permutations, entry_count * n, 0, n),
                )
            )
        sord = r.column(b"SORD", "I")
        words = [sord[k::4] for k in range(4)]
        n_targets = r.count(b"STGT")
        orderings = SwitchOrderings(
            array("q", words[0]),
            array("q", (w >> 16 & 0xFF for w in words[3])),
            array("q", (w >> 24 for w in words[3])),
            array("d", map(double, words[2])),
            array("q", (w - (1 << 32) if w >= 1 << 31 else w for w in words[1])),
            array("q", (w & 0xFFFF for w in words[3])),
        )
        _range("ordering evaluation", orderings.evaluation, 0, n_eval)
        written = {e for call in calls for e in call.evaluations}
        if not written.issuperset(orderings.evaluation):
            raise ValueError(
                "reweight data: an ordering reads an evaluation no call writes"
            )
        _range("ordering target", orderings.target, -1, n_targets)
        _range("ordering parton density", orderings.pdf0, 0, PDF_COLUMNS)
        _range("ordering parton density", orderings.pdf1, 0, PDF_COLUMNS)
        flow_rows = r.rows(b"SFLW")
        sfid = r.column(b"SFID", "I")
        if len(flow_rows) != n_ref:
            raise ValueError("reweight data: flow table count mismatch")
        flows = []
        for first, count in flow_rows:
            if first + count > len(sfid):
                raise ValueError("reweight data: flow ids out of range")
            flows.append(tuple(text(i, True) for i in sfid[first : first + count]))
        stgt = r.column(b"STGT", "I")
        w = [stgt[k::8] for k in range(8)]
        targets = SwitchTargets(
            array("q"), array("q"), [], [], [], array("d"), array("q"), []
        )
        for k in range(n_targets):
            evaluation, process = w[0][k], w[1][k]
            if evaluation >= n_eval or process >= n_tgt:
                raise ValueError("reweight data: switch target out of range")
            width = len(flows[evaluations[evaluation].process])
            assignment = w[5][k] - (1 << 32) if w[5][k] >= 1 << 31 else w[5][k]
            if assignment >= 0 and w[7][k] == NULL:
                raise ValueError("reweight data: assignment without its process id")
            targets.evaluation.append(evaluation)
            targets.target_process.append(process)
            targets.colors.append(ints(w[2][k], width, nullable=False))
            targets.flows.append(ints(w[3][k], width, 0, width, False))
            targets.legs.append(ints(w[4][k], n, 0, n))
            targets.factor.append(double(w[6][k]))
            targets.assignment.append(assignment)
            targets.assignment_key.append(text(w[7][k]))
        helicities = tuple(r.column(b"SHEL", "i"))
        if len(helicities) != exported * n:
            raise ValueError("reweight data: switch helicity count mismatch")
        switch = SwitchRows(
            evaluations,
            calls,
            orderings,
            targets,
            flows,
            helicities,
            arrangements,
        )
    return ReweightTables(
        provider,
        processes[:n_ref],
        processes[n_ref:],
        channels,
        switch,
    )


def reweight_data_header(
    tables: ReweightTables, file_name: str, sha256_hex: str
) -> str:
    """The generated header of a provider: the constants the library needs
    before its data file is loaded."""
    p = tables.provider
    return "\n".join(
        [
            "/* SPDX-License-Identifier: 0BSD */",
            "/* Generated with the data file it names. */",
            f'#define REWEIGHT_PROVIDER_ID "{p.id}"',
            f"#define REWEIGHT_PARTICLE_COUNT {p.particle_count}u",
            f"#define REWEIGHT_EXPORTED_HELICITY_COUNT {p.exported_helicity_count}u",
            f"#define UMAMI_HAS_ALPHA_S {int(p.supports_alpha_s)}",
            f'#define REWEIGHT_DATA_FILE "{file_name}"',
            f'#define REWEIGHT_DATA_SHA256 "{sha256_hex}"',
            "",
        ]
    )


def write_reweight_data(output: Path, tables: ReweightTables) -> Path:
    """Write ``reweight_<id>.rwd`` (replaced atomically) and its header
    ``reweight_<id>.h`` to ``output``; returns the data file."""
    output = Path(output)
    identifier = tables.provider.id
    header, chunks = _encode(tables)
    path = output / f"reweight_{identifier}.rwd"
    temporary = output / f".reweight_{identifier}.rwd.{os.getpid()}.tmp"
    try:
        with open(temporary, "wb") as stream:
            stream.write(header)
            for chunk in chunks:
                stream.write(chunk)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    (output / f"reweight_{identifier}.h").write_text(
        reweight_data_header(tables, path.name, content_sha256(header)), encoding="utf-8"
    )
    return path


__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "FlavourRow",
    "ProcessRow",
    "ProviderScalars",
    "ReweightTables",
    "SwitchCall",
    "SwitchEvaluation",
    "SwitchOrderings",
    "SwitchRows",
    "SwitchTargets",
    "content_sha256",
    "decode_reweight_data",
    "encode_reweight_data",
    "reseal_reweight_data",
    "reweight_data_header",
    "section_table",
    "write_reweight_data",
]
