# SPDX-License-Identifier: 0BSD
"""Process-group schema 2: crossed flavour-union members (WP-E, T12, T16)."""

from __future__ import annotations

import io
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

from pyamplicol.api.errors import ArtifactError
from pyamplicol.artifacts.process_groups import validate_process_groups
from pyamplicol.artifacts.umami_reweight import (
    switch_catalogue,
    write_reweight_provider,
)
from pyamplicol.generation.process_grouping import (
    PROCESS_GROUPS_CROSSING_SCHEMA_VERSION,
    PROCESS_GROUPS_SCHEMA_VERSION,
    GroupingBasis,
    GroupMember,
    ProcessGroupMap,
    crossing_signs,
)

TEMPLATE = GroupMember(
    process_id="u_g_to_u_g",
    expression="u g > u g",
    pdgs=(2, 21, 2, 21),
    initial_count=2,
    permutation=(0, 1, 2, 3),
)
# g g > u u~ through u g > u g: template slot 0 (incoming u) takes the outgoing
# u~ (leg 3), slot 3 (outgoing g) the incoming gluon on leg 1.
CROSSED = GroupMember(
    process_id="g_g_to_u_ubar",
    expression="g g > u u~",
    pdgs=(21, 21, 2, -2),
    initial_count=2,
    permutation=(3, 0, 2, 1),
    basis=GroupingBasis.FLAVOUR_UNION,
    generated=False,
    patterns=(0,),
    pattern_signs=(1,),
    crossing_momentum_signs=(-1, 1, 1, -1),
    crossing_helicity_signs=(-1, 1, 1, 1),
)
# d g > d g through u g > u g, on its own sides
UNCROSSED = GroupMember(
    process_id="d_g_to_d_g",
    expression="d g > d g",
    pdgs=(1, 21, 1, 21),
    initial_count=2,
    permutation=(0, 1, 2, 3),
    basis=GroupingBasis.FLAVOUR_UNION,
    generated=False,
    patterns=(0,),
    pattern_signs=(1,),
)
PAIRS = (((2, 0),),)


def _map(*members: GroupMember) -> ProcessGroupMap:
    group_map = ProcessGroupMap()
    group_map.add_group([TEMPLATE, *members], PAIRS)
    return group_map


def _fields(member: GroupMember) -> dict[str, Any]:
    return {item.name: getattr(member, item.name) for item in fields(member)}


def test_crossing_signs_follow_the_sides() -> None:
    assert crossing_signs(CROSSED.permutation, 2, 2) == (-1, 1, 1, -1)
    assert crossing_signs((1, 0, 3, 2), 2, 2) == (1, 1, 1, 1)
    assert CROSSED.crossed and not UNCROSSED.crossed and not TEMPLATE.crossed


def test_schema_2_round_trips_and_is_written_only_with_a_crossed_member() -> None:
    document = _map(CROSSED, UNCROSSED).as_document()
    assert document["schema_version"] == PROCESS_GROUPS_CROSSING_SCHEMA_VERSION == 2
    records = document["groups"][0]["members"]  # type: ignore[index]
    assert records[1]["crossing_momentum_signs"] == [-1, 1, 1, -1]
    assert records[1]["crossing_helicity_signs"] == [-1, 1, 1, 1]
    assert "crossing_momentum_signs" not in records[2]
    restored = ProcessGroupMap.from_json(_map(CROSSED, UNCROSSED).to_json())
    assert list(restored.members()) == [TEMPLATE, CROSSED, UNCROSSED]
    assert restored.to_json() == _map(CROSSED, UNCROSSED).to_json()
    validate_process_groups(restored, [TEMPLATE.process_id])

    plain = _map(UNCROSSED)
    assert plain.as_document()["schema_version"] == PROCESS_GROUPS_SCHEMA_VERSION
    assert "crossing" not in plain.to_json()
    assert GroupMember.from_dict(CROSSED.as_dict()) == CROSSED


def test_readers_refuse_crossing_under_schema_1_and_unknown_schemas() -> None:
    document = json.loads(_map(CROSSED).to_json())
    for version in (1,):
        with pytest.raises(ValueError, match="schema version 1 cannot carry"):
            ProcessGroupMap.from_document({**document, "schema_version": version})
    for version in (0, 3, True, "2", None):
        with pytest.raises(ValueError, match="unsupported"):
            ProcessGroupMap.from_document({**document, "schema_version": version})
    # one sign vector alone is refused under schema 1 too
    record = document["groups"][0]["members"][1]
    del record["crossing_momentum_signs"]
    with pytest.raises(ValueError, match="schema version 1"):
        ProcessGroupMap.from_document({**document, "schema_version": 1})
    # a plain schema-2 map without a crossed member still reads
    plain = json.loads(_map(UNCROSSED).to_json())
    assert ProcessGroupMap.from_document({**plain, "schema_version": 2}).groups


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"crossing_helicity_signs": None}, "both crossing sign vectors"),
        ({"crossing_momentum_signs": None}, "both crossing sign vectors"),
        ({"patterns": None, "pattern_signs": None}, "not a flavour-union member"),
        ({"crossing_helicity_signs": (-1, 1, 1)}, "one crossing sign per leg"),
        ({"crossing_momentum_signs": (-2, 1, 1, -1)}, "other than"),
        (
            {
                "crossing_momentum_signs": (1, 1, 1, 1),
                "crossing_helicity_signs": (1, 1, 1, 1),
            },
            "is not crossed",
        ),
        ({"conjugated": True}, "both crossed and conjugated"),
        ({"crossing_helicity_signs": (-1, -1, 1, 1)}, "uncrossed slot"),
    ],
)
def test_invalid_crossing_members_are_refused(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GroupMember(**{**_fields(CROSSED), **change})


def test_crossing_signs_must_match_the_permutation() -> None:
    wrong = replace(
        CROSSED,
        crossing_momentum_signs=(1, -1, 1, -1),
        crossing_helicity_signs=(1, 1, 1, 1),
    )
    with pytest.raises(ValueError, match="disagree with its permutation"):
        _map(wrong)
    document = json.loads(_map(CROSSED).to_json())
    document["groups"][0]["members"][1]["crossing_momentum_signs"] = [1, -1, 1, -1]
    document["groups"][0]["members"][1]["crossing_helicity_signs"] = [1, 1, 1, 1]
    with pytest.raises(ValueError, match="disagree"):
        ProcessGroupMap.from_document(document)


def test_validation_refuses_side_crossing_without_union_signs() -> None:
    unsigned = replace(
        CROSSED, crossing_momentum_signs=None, crossing_helicity_signs=None
    )
    blind = replace(
        unsigned, basis=GroupingBasis.FLAVOR_BLIND, patterns=None, pattern_signs=None
    )
    for member in (unsigned, blind):
        group_map = ProcessGroupMap()
        group_map.add_group([TEMPLATE, member], PAIRS if member.patterns else None)
        with pytest.raises(ArtifactError, match="across the '>' boundary"):
            validate_process_groups(group_map, [TEMPLATE.process_id])
    # a within-side swap of the initial legs needs no signs
    swapped = replace(blind, expression="g g > u~ u", permutation=(1, 0, 3, 2))
    group_map = ProcessGroupMap()
    group_map.add_group([TEMPLATE, swapped])
    validate_process_groups(group_map, [TEMPLATE.process_id])
    broken = ProcessGroupMap()
    broken.add_group([TEMPLATE, replace(swapped, permutation=(0, 1, 2))])
    with pytest.raises(ArtifactError, match="not a leg permutation"):
        validate_process_groups(broken, [TEMPLATE.process_id])


def _reweight_artifact(path: Path) -> Path:
    root = path / "API" / "umami"
    root.mkdir(parents=True)
    document = {
        "kind": "pyamplicol-umami-metadata",
        "schema_version": 4,
        "provider": {"id": "p0", "color_accuracy": "lc", "particle_count": 4},
    }
    (root / "metadata.json").write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kind": "something-else"}, "not a pyAmpliCol process-group map"),
        ({"schema_version": 7}, "unsupported"),
        ({"schema_version": 1}, "schema version 1 cannot carry"),
    ],
)
def test_reweight_writer_validates_the_raw_process_groups(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    lc = _reweight_artifact(tmp_path / "lc")
    full = _reweight_artifact(tmp_path / "full")
    document = {**json.loads(_map(CROSSED).to_json()), **change}
    (lc / "process-groups.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        write_reweight_provider(lc, full, tmp_path / "out", switch_lines=2)


def test_flavour_switch_checks_crossed_members() -> None:
    reference = {
        "provider": {"incoming_count": 2, "masses": [0.0] * 4},
        "runtime_processes": [{"id": TEMPLATE.process_id, "pdgs": list(TEMPLATE.pdgs)}],
        "pdg_ids": [list(TEMPLATE.pdgs), list(CROSSED.pdgs)],
        "channels": [],
        "color_flows": [],
    }
    document = _map(CROSSED).as_document()
    catalogue = switch_catalogue(reference, document, 1)
    crossed = [e for e in catalogue["evaluations"] if "signs" in e]
    assert crossed and all(
        e["assignment_key"] == "g_g_to_u_ubar"
        and e["helicity_signs"] == list(CROSSED.crossing_helicity_signs)
        for e in crossed
    )
    # sigma is re-derived from the legs: a tampered record is refused
    (member,) = [
        m for m in document["groups"][0]["members"] if "crossing_momentum_signs" in m
    ]
    member["crossing_momentum_signs"] = [1] * len(member["crossing_momentum_signs"])
    with pytest.raises(ValueError, match="crossed member 'g_g_to_u_ubar'"):
        switch_catalogue(reference, document, 1)


def test_inspection_reports_crossing(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyamplicol.artifacts.inspection as inspection

    group_map = _map(CROSSED, UNCROSSED)
    monkeypatch.setattr(inspection, "read_process_groups", lambda _path: group_map)
    stored = inspection.stored_process_groups("artifact")
    assert stored is not None
    crossed, uncrossed = stored.groups[0].members[1:]
    assert crossed.crossing_momentum_signs == (-1, 1, 1, -1)
    assert crossed.crossing_helicity_signs == (-1, 1, 1, 1)
    assert uncrossed.crossing_momentum_signs is None
    selection = inspection.process_group_selection("artifact", CROSSED.process_id)
    assert selection is not None
    assert selection.crossed and selection.crossing_momentum_signs == (-1, 1, 1, -1)
    other = inspection.process_group_selection("artifact", UNCROSSED.process_id)
    assert other is not None and not other.crossed


def test_inspect_selector_names_a_crossed_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyamplicol.artifacts as artifact_module
    import pyamplicol.artifacts.inspection as inspection
    from pyamplicol.cli import run_cli

    group_map = _map(CROSSED, UNCROSSED)
    monkeypatch.setattr(inspection, "read_process_groups", lambda _path: group_map)
    groups = inspection.stored_process_groups("artifact")
    fake = type("Inspection", (), {"processes": (), "process_groups": groups})()
    monkeypatch.setattr(artifact_module, "inspect_artifact", lambda _path: fake)
    stderr = io.StringIO()
    status = run_cli(
        ("inspect", str(tmp_path), "--process", "g g > u u~", "--progress", "off"),
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert status == 2
    message = " ".join(stderr.getvalue().split())
    assert "is a crossed flavour-union member of process group u_g_to_u_g" in message
    assert "template's assignments" in message
