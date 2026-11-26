# SPDX-License-Identifier: 0BSD
"""The data file of the reweight providers: round trip, determinism, and the
checks the C loader repeats."""

from __future__ import annotations

import dataclasses
import struct
from array import array

import pytest

from pyamplicol.artifacts.umami_reweight_data import (
    FlavourRow,
    ProcessRow,
    ProviderScalars,
    ReweightTables,
    SwitchCall,
    SwitchEvaluation,
    SwitchOrderings,
    SwitchRows,
    SwitchTargets,
    content_sha256,
    decode_reweight_data,
    encode_reweight_data,
    reseal_reweight_data,
    reweight_data_header,
    section_table,
    write_reweight_data,
)

N = 4


def _flavour(**changes) -> FlavourRow:
    row = dict(
        reference_process=0,
        target_process=0,
        permutation=(0, 1, 2, 3),
        reference_signs=None,
        reference_helicity=(0, -1),
        target_helicity=(-1, 1),
        reference_assignment=-1,
        assignment_key=None,
        target_permutation=None,
        target_signs=None,
        target_factor=1.0,
        target_assignment=-1,
        target_assignment_key=None,
        switch_arrangement=0,
    )
    row.update(changes)
    return FlavourRow(**row)


def _tables(switch: bool = True) -> ReweightTables:
    provider = ProviderScalars("p0", N, 2, "../lc", "../full", "lc-id", None, True)
    references = [
        ProcessRow("gg", (21, 21, 21, 21), "aS", 2, 2),
        ProcessRow("ug", (2, 21, 2, 21), "aS", 2, 2),
    ]
    targets = [ProcessRow("full_gg", (21, 21, 21, 21), "aS", 2, 3)]
    channels = [
        [
            _flavour(),
            _flavour(
                reference_process=1,
                reference_signs=(1, -1, -1, 1),
                reference_assignment=1,
                assignment_key="u_g_to_u_g",
                target_permutation=(1, 0, 2, 3),
                target_factor=0.5,
                target_assignment=0,
                target_assignment_key="full_gg",
                switch_arrangement=1,
            ),
        ],
        [_flavour(permutation=(1, 0, 3, 2))],
    ]
    rows = None
    if switch:
        evaluations = [
            SwitchEvaluation(0, (0, 1, 2, 3), -1, None, None, None),
            SwitchEvaluation(
                1, (1, 0, 2, 3), 2, "d_g_to_d_g", (1, -1, -1, 1), (1, 1, -1, 1)
            ),
        ]
        calls = [
            SwitchCall((0,), None, None, None),
            SwitchCall((1,), (2,), (0,), (0, 1, 3, 2)),
        ]
        orderings = SwitchOrderings(
            array("q", [0, 1, 1]),
            array("q", [6, 8, 6]),
            array("q", [6, 6, 4]),
            array("d", [0.25, 1 / 3, 0.25]),
            array("q", [-1, 0, 1]),
            array("q", [0, 1, 1]),
        )
        colors = (3, 4)
        targets_ = SwitchTargets(
            array("q", [1, 1]),
            array("q", [0, 0]),
            [colors, (5, 6)],
            [(1, 0), (0, 1)],
            [None, (0, 1, 3, 2)],
            array("d", [1.0, 2.0]),
            array("q", [-1, 0]),
            [None, "full_gg"],
        )
        rows = SwitchRows(
            evaluations,
            calls,
            orderings,
            targets_,
            [(), ("flow:a", "flow:b")],
            tuple(range(2 * N)),
            2,
        )
    return ReweightTables(provider, references, targets, channels, rows)


@pytest.mark.parametrize("switch", [False, True])
def test_round_trip_and_determinism(switch):
    tables = _tables(switch)
    data = encode_reweight_data(tables)
    assert data == encode_reweight_data(_tables(switch))
    assert decode_reweight_data(data) == tables
    header = reweight_data_header(tables, "reweight_p0.rwd", content_sha256(data))
    assert f'#define REWEIGHT_DATA_SHA256 "{content_sha256(data)}"' in header
    assert "#define REWEIGHT_PARTICLE_COUNT 4u" in header


def test_equal_arrays_are_stored_once():
    tables = _tables()
    data = encode_reweight_data(tables)
    pool = section_table(data)["I32P"][1]
    # one more flavour with every array equal to the first's: no pool growth
    tables.channels[1].append(_flavour())
    assert section_table(encode_reweight_data(tables))["I32P"][1] == pool


def _patch(data: bytes, tag: str, row: int, field: int, value: int, signed=False):
    offset, count, size = section_table(data)[tag]
    assert row < count
    patched = bytearray(data)
    struct.pack_into(
        "<i" if signed else "<I", patched, offset + row * size + 4 * field, value
    )
    return reseal_reweight_data(bytes(patched))


def _pool_count(data: bytes) -> int:
    return section_table(data)["I32P"][1]


CORRUPTIONS = {
    # flavour 0 (FLAV fields: 2 = permutation ref, 8 = reference_assignment)
    "flavour process": (lambda d: _patch(d, "FLAV", 0, 0, 7), "flavour"),
    "flavour permutation": (
        lambda d: _patch(d, "FLAV", 0, 2, _pool_count(d) - 1),
        "flavour|array",
    ),
    "assignment without key": (
        lambda d: _patch(d, "FLAV", 0, 8, 0, True),
        "process id|flavour",
    ),
    "call without evaluations": (lambda d: _patch(d, "SCAL", 0, 0, 0), "call"),
    "ordering target": (lambda d: _patch(d, "SORD", 0, 1, 2, True), "ordering target"),
    "target evaluation": (lambda d: _patch(d, "STGT", 0, 0, 2), "target"),
    "target colours": (lambda d: _patch(d, "STGT", 0, 2, _pool_count(d) - 1), "array"),
    "target flows": (lambda d: _patch(d, "STGT", 0, 3, 0), "array value"),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_semantic_corruption_is_refused(name):
    patch, message = CORRUPTIONS[name]
    with pytest.raises(ValueError, match=message):
        decode_reweight_data(patch(encode_reweight_data(_tables())))


@pytest.mark.parametrize(
    "calls, message",
    [
        # switch_totals writes only the first evaluation of a plain call
        ([SwitchCall((0, 1), None, None, None)], "malformed plain switch call"),
        # nothing writes evaluation 1, which orderings 1 and 2 read
        ([SwitchCall((0,), None, None, None)], "no call writes"),
    ],
)
def test_evaluations_no_call_writes_are_refused(calls, message):
    tables = _tables()
    tables = dataclasses.replace(
        tables, switch=dataclasses.replace(tables.switch, calls=calls)
    )
    with pytest.raises(ValueError, match=message):
        decode_reweight_data(encode_reweight_data(tables))


def test_damaged_files_are_refused():
    data = encode_reweight_data(_tables())
    flipped = bytearray(data)
    flipped[-3] ^= 0x10
    with pytest.raises(ValueError, match="checksum"):
        decode_reweight_data(bytes(flipped))
    with pytest.raises(ValueError, match="size"):
        decode_reweight_data(data[:-8])
    newer = bytearray(data)
    struct.pack_into("<I", newer, 8, 2)
    with pytest.raises(ValueError, match="format version 2"):
        decode_reweight_data(bytes(newer))
    with pytest.raises(ValueError, match="not a reweight data file"):
        decode_reweight_data(b"\0" * len(data))


def test_strings_with_nul_are_refused():
    tables = _tables()
    tables.channels[0][1] = _flavour(reference_assignment=0, assignment_key="a\0b")
    with pytest.raises(ValueError, match="NUL"):
        encode_reweight_data(tables)


def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    import pyamplicol.artifacts.umami_reweight_data as module

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_reweight_data(tmp_path, _tables())
    assert list(tmp_path.iterdir()) == []
