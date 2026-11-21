# SPDX-License-Identifier: 0BSD
"""Crafted UMAMI metadata must not reach the reweight provider data."""

from __future__ import annotations

import json
from array import array
from pathlib import Path

import pytest

from pyamplicol.artifacts.umami_reweight import (
    SwitchTargetTable,
    load_umami_metadata,
    reweight_tables,
    switch_catalogue,
    write_reweight_provider,
    write_switch_sidecar,
)


def _process(identifier="proc"):
    return {
        "id": identifier,
        "pdgs": [21, 21, 21, 21],
        "alpha_s_parameter": "aS",
        "helicity_ids": ["h0"],
    }


def _document(provider_id="p0", permutation=(0, 1, 2, 3), process_index=0):
    return {
        "kind": "pyamplicol-umami-metadata",
        "schema_version": 4,
        "provider": {
            "id": provider_id,
            "particle_count": 4,
            "color_accuracy": "lc",
            "artifact_id": "a",
        },
        "runtime_processes": [_process()],
        "helicities": [[1, 1, 1, 1]],
        "channels": [
            {
                "processes": [
                    {
                        "runtime": {
                            "process_index": process_index,
                            "permutation": list(permutation),
                            "helicity_export": [0],
                        }
                    }
                ]
            }
        ],
    }


def _write(artifact: Path, document) -> Path:
    root = artifact / "API" / "umami"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(document), encoding="utf-8")
    return artifact


def test_valid_metadata_gives_tables():
    tables = reweight_tables(_document(), [_document()], "lc", "full")
    (row,) = tables.channels[0]
    assert row.permutation == (0, 1, 2, 3) and row.target_permutation is None


@pytest.mark.parametrize(
    "document",
    [
        _document(permutation=(0, 1, 2, "3}; void f(void) {}")),
        _document(permutation=(0, 1, 2, True)),
        _document(process_index="0"),
    ],
)
def test_non_integer_metadata_is_rejected(document):
    with pytest.raises(ValueError, match="integer"):
        reweight_tables(document, [_document()], "lc", "full")


def test_provider_id_is_validated(tmp_path):
    lc = _write(tmp_path / "lc", _document(provider_id="$(shell touch PWNED)"))
    full = _write(tmp_path / "full", _document())
    with pytest.raises(ValueError, match="provider id"):
        write_reweight_provider(lc, full, tmp_path / "out")
    assert not (tmp_path / "out" / "providers.mk").exists()


def test_metadata_path_stays_in_artifact(tmp_path):
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    index = {
        "kind": "pyamplicol-umami-provider-index",
        "providers": [{"id": "p0", "metadata": "../../../secret.json"}],
    }
    artifact = _write(tmp_path / "lc", index)
    with pytest.raises(ValueError, match="leaves"):
        load_umami_metadata(artifact)


def _switch_reference(rows, masses):
    """A provider of the given flavours (rows in slot order) and leg masses."""
    return {
        "provider": {"incoming_count": 2, "masses": list(masses)},
        "runtime_processes": [
            {"id": f"r{index}", "pdgs": list(row)} for index, row in enumerate(rows)
        ],
        "pdg_ids": [list(row) for row in rows],
        "channels": [],
        "color_flows": [],
    }


TTJ_ROWS = (
    [21, 21, 6, -6, 21], [2, -2, 6, -6, 21], [21, 2, 6, -6, 2], [1, -1, 6, -6, 21]
)


def _moved(catalogue, reference, heavy):
    """Targets and evaluations with a pdg of ``heavy`` outside the massive slots."""
    masses = reference["provider"]["masses"]
    slots = {slot for slot, mass in enumerate(masses) if mass > 0}
    rows = list(catalogue["target_pdgs"])
    for e in catalogue["evaluations"]:
        pdgs = reference["runtime_processes"][e["process"]]["pdgs"]
        rows.append([pdg for _, pdg in sorted(zip(e["legs"], pdgs, strict=True))])
    return [r for r in rows if {s for s, p in enumerate(r) if abs(p) in heavy} - slots]


def test_switch_never_moves_massive_quarks():
    reference = _switch_reference(TTJ_ROWS, [0.0, 0.0, 173.0, 173.0, 0.0])
    catalogue = switch_catalogue(reference, None, 2)
    assert catalogue["target_pdgs"]
    assert _moved(catalogue, reference, {6}) == []
    # one ordering per flavour and beam orientation: no jet slot to exchange with
    assert len(catalogue["orderings"]) == 7


def _arrangement_targets(catalogue):
    """Per arrangement, the targets of its orderings and its weight per pdf pair."""
    targets, totals = {}, {}
    for ordering in catalogue["orderings"]:
        k = ordering["arrangement"]
        if ordering["target"] >= 0:
            pdgs = tuple(catalogue["target_pdgs"][ordering["target"]])
            targets.setdefault(k, set()).add(pdgs)
        key = (k, tuple(ordering["pdf"]))
        totals[key] = totals.get(key, 0.0) + ordering["weight"]
    return targets, totals


@pytest.mark.parametrize(
    ("rows", "masses", "lines"),
    [
        # a conjugated flavour has t~ t in the slots of t t~
        ((*TTJ_ROWS, [-2, 2, -6, 6, 21]), [0.0, 0.0, 173.0, 173.0, 0.0], 2),
        # charged leptons (fixed, massless) of a conjugated flavour
        (([2, -2, 11, -11, 21], [21, 2, 11, -11, 2], [-2, 2, -11, 11, 21]), [0] * 5, 1),
    ],
)
def test_switch_keeps_the_event_arrangement_of_non_partons(rows, masses, lines):
    slots = (2, 3)
    reference = _switch_reference(rows, masses)
    catalogue = switch_catalogue(reference, None, lines)
    assert catalogue["arrangement_count"] == 2
    targets, totals = _arrangement_targets(catalogue)
    for index, row in enumerate(rows):
        k = catalogue["row_arrangements"][index]
        # every target of the event's arrangement has its particles where it has them
        assert targets[k]
        assert {tuple(t[s] for s in slots) for t in targets[k]} == {
            tuple(row[s] for s in slots)
        }
    for k in range(2):
        evaluations = {
            o["evaluation"] for o in catalogue["orderings"] if o["arrangement"] == k
        }
        row = rows[catalogue["row_arrangements"].index(k)]
        expected = {tuple(row[s] for s in slots)}
        for e in evaluations:
            e = catalogue["evaluations"][e]
            pdgs = reference["runtime_processes"][e["process"]]["pdgs"]
            at = dict(zip(e["legs"], pdgs, strict=True))
            assert {tuple(at[s] for s in slots)} == expected
    # each flavour and beam orientation keeps its weight in every arrangement
    assert totals == pytest.approx(dict.fromkeys(totals, 1.0))


def test_switch_of_one_arrangement_has_one_table():
    reference = _switch_reference(TTJ_ROWS, [0.0, 0.0, 173.0, 173.0, 0.0])
    catalogue = switch_catalogue(reference, None, 2)
    assert catalogue["arrangement_count"] == 1
    assert {o["arrangement"] for o in catalogue["orderings"]} == {0}


@pytest.mark.parametrize("b_mass", [0.0, 4.7])
def test_switch_moves_b_quarks_only_when_massless(b_mass):
    rows = ([21, 21, 5, -5, 21], [2, -2, 5, -5, 21], [21, 2, 5, -5, 2])
    reference = _switch_reference(rows, [0.0, 0.0, b_mass, b_mass, 0.0])
    catalogue = switch_catalogue(reference, None, 2)
    if b_mass:
        assert _moved(catalogue, reference, {5}) == []
    else:
        assert any(row[4] in (5, -5) for row in catalogue["target_pdgs"])


def test_switch_rejects_partons_with_massless_and_massive_slots():
    reference = _switch_reference(
        ([21, 21, 5, -5, 21], [21, 21, 21, 5, -5]), [0.0, 0.0, 4.7, 4.7, 0.0]
    )
    with pytest.raises(ValueError, match="massless and massive slots"):
        switch_catalogue(reference, None, 2)


@pytest.mark.parametrize(
    ("version", "signs", "message"),
    [
        (3, None, "unsupported UMAMI metadata schema version 3"),
        (None, None, "unsupported UMAMI metadata schema version None"),
        (4, [-1, 1, 1, -1], "schema 4 cannot carry momentum signs"),
        (5, [-1, 1, 1, -1], None),
        (5, None, None),
    ],
)
def test_metadata_schema_versions(tmp_path, version, signs, message):
    document = _document()
    if version is None:
        del document["schema_version"]
    else:
        document["schema_version"] = version
    if signs is not None:
        document["channels"][0]["processes"][0]["runtime"]["momentum_signs"] = signs
    artifact = _write(tmp_path / "a", document)
    if message is None:
        assert load_umami_metadata(artifact) == [document]
    else:
        with pytest.raises(ValueError, match=message):
            load_umami_metadata(artifact)


def test_reweight_refuses_crossed_targets(tmp_path):
    crossed = _document()
    crossed["schema_version"] = 5
    crossed["channels"][0]["processes"][0]["runtime"]["momentum_signs"] = [-1, 1, 1, -1]
    full = _write(tmp_path / "full", crossed)
    lc = _write(tmp_path / "lc", _document())
    with pytest.raises(ValueError, match="reweight targets with crossed"):
        write_reweight_provider(lc, full, tmp_path / "out")


def _crossed_reference(signs):
    """A union reference whose flavour is assignment 1 with momentum signs, and a
    full-colour target of the member's own process."""
    reference = _document()
    reference["schema_version"] = 5
    reference["provider"]["incoming_count"] = 2
    reference["pdg_ids"] = [[2, 21, 2, 21]]
    entry = reference["channels"][0]["processes"][0]
    entry["matrix_elements"] = [{"pdg_ids": 0}]
    entry["runtime"].update(
        assignment=1, assignment_key="u_g_to_u_g", momentum_signs=signs
    )
    target = _document()
    target["runtime_processes"] = [
        {
            **_process("u_g_to_u_g"),
            "pdgs": [2, 21, 2, 21],
            "helicity_ids": ["h:1,1,1,1"],
        }
    ]
    target["channels"][0]["processes"][0]["members"] = [
        {
            "process_id": "u_g_to_u_g",
            "pdgs": [2, 21, 2, 21],
            "runtime": {"process_index": 0, "permutation": [0, 1, 2, 3], "factor": 1.0},
        }
    ]
    return reference, target


def test_crossed_reference_rows_carry_their_signs():
    reference, target = _crossed_reference([1, -1, -1, 1])
    (row,) = reweight_tables(reference, [target], "lc", "full").channels[0]
    assert row.reference_signs == (1, -1, -1, 1)
    assert (row.reference_process, row.target_process) == (0, 0)
    assert (row.reference_assignment, row.assignment_key) == (1, "u_g_to_u_g")
    assert row.target_permutation is not None and row.target_signs is None
    assert (row.target_factor, row.target_assignment, row.target_assignment_key) == (
        1.0, -1, None
    )
    # uncrossed rows: no signs on either side
    (plain,) = reweight_tables(_document(), [_document()], "lc", "full").channels[0]
    assert plain.reference_signs is None and plain.target_signs is None
    for signs in ([1, -1, 2, 1], [1, -1, True, 1]):
        bad, _ = _crossed_reference(signs)
        with pytest.raises(ValueError, match=r"momentum signs|integer"):
            reweight_tables(bad, [target], "lc", "full")


def test_canonical_legs_keep_crossed_slots():
    from pyamplicol.artifacts.umami_reweight import _canonical_legs

    # identical outgoing gluons are sorted, a crossed one (incoming slot 1) is not
    assert _canonical_legs([2, -2, 21, 21], [0, 1, 3, 2], 2) == (0, 1, 2, 3)
    assert _canonical_legs([2, 2, 21, 21], [0, 3, 2, 1], 2) == (0, 3, 2, 1)
    assert _canonical_legs([2, 2, 21, 21, 21], [0, 4, 3, 1, 2], 2) == (0, 4, 2, 1, 3)
    # a template-initial slot reading an outgoing gluon joins the outgoing ones
    assert _canonical_legs([21, 2, 21, 2, 21], [4, 1, 2, 3, 0], 2) == (2, 1, 4, 3, 0)


def _record(identifier, pdgs, permutation, signs=None, rho=None):
    record = {
        "process_id": identifier,
        "pdgs": list(pdgs),
        "initial_count": 2,
        "permutation": list(permutation),
        "conjugated": False,
        "factor": {"numerator": 1, "denominator": 1},
    }
    if signs is not None:
        record.update(
            flavour_union_patterns=[0],
            flavour_union_pattern_signs=[1],
            crossing_momentum_signs=list(signs),
        )
        if rho is not None:
            record["crossing_helicity_signs"] = list(rho)
    return record


def _crossed_groups(signs=(1, -1, -1, 1), rho=(1, -1, 1, 1)):
    # template u u~ > g g; crossed member u g > u g (slots 1 and 2 cross)
    return {
        "groups": [
            {
                "representative": {"process_id": "r0"},
                "members": [
                    _record("r0", [2, -2, 21, 21], [0, 1, 2, 3]),
                    _record("u_g_to_u_g", [2, 21, 2, 21], [0, 2, 1, 3], signs, rho),
                ],
            }
        ]
    }


def test_switch_evaluations_of_crossed_members_carry_their_signs():
    reference = _switch_reference(([2, -2, 21, 21], [2, 21, 2, 21]), [0.0] * 4)
    reference["runtime_processes"] = reference["runtime_processes"][:1]
    catalogue = switch_catalogue(reference, _crossed_groups(), 1)
    crossed = [e for e in catalogue["evaluations"] if e["assignment"] == 1]
    plain = [e for e in catalogue["evaluations"] if e["assignment"] == 0]
    assert crossed and plain
    assert all("signs" not in e for e in plain)
    for evaluation in crossed:
        legs = evaluation["legs"]
        assert evaluation["signs"] == [
            -1 if (r < 2) != (slot < 2) else 1 for r, slot in enumerate(legs)
        ]
        assert evaluation["signs"] == [1, -1, -1, 1]
        assert evaluation["helicity_signs"] == [1, -1, 1, 1]
    # one native call per crossing: never a crossed and an uncrossed request together
    for call in catalogue["calls"]:
        kinds = {
            tuple(catalogue["evaluations"][e].get("signs") or ())
            for e in call["evaluations"]
        }
        assert len(kinds) == 1
    from pyamplicol.artifacts.umami_reweight import _switch_rows

    reference["helicities"] = [[1, 1, 1, 1]]
    rows = _switch_rows(reference, catalogue, lambda *_: (0, None, 1.0, -1, None))
    index = catalogue["evaluations"].index(crossed[0])
    assert rows.evaluations[index].signs == (1, -1, -1, 1)
    assert rows.evaluations[index].helicity_signs == (1, -1, 1, 1)
    assert any(
        e.assignment_key == "r0" and e.signs is None and e.helicity_signs is None
        for e in rows.evaluations
    )


@pytest.mark.parametrize(
    ("signs", "rho"),
    [
        ((1, 1, -1, 1), (1, 1, 1, 1)),
        ((1, -1, -1, 1), (1, -1, 1, -1)),
        ((1, -1, -1, 1), None),
    ],
)
def test_switch_refuses_inconsistent_crossings(signs, rho):
    reference = _switch_reference(([2, -2, 21, 21], [2, 21, 2, 21]), [0.0] * 4)
    reference["runtime_processes"] = reference["runtime_processes"][:1]
    with pytest.raises(ValueError, match="inconsistent crossing signs"):
        switch_catalogue(reference, _crossed_groups(signs, rho), 1)


@pytest.mark.parametrize("count", [0, 1, 30_000])
def test_switch_sidecar_has_the_bytes_of_json_dumps(tmp_path, count):
    # chunked writing over several chunks, negative pdg ids, an empty table
    width = 5
    pdgs = [[21, -2, 2 + k % 3, -(k % 4) - 1, 21] for k in range(count)]
    flows = [[[501 + k % 7, 0], [0, 502], [503, 501], [0, 0], [502, 503]]
             for k in range(count // 2)]
    targets = SwitchTargetTable(
        array("i"), array("i"), [], array("i", [v for row in pdgs for v in row]), width
    )
    catalogue = {
        "switch_lines": 3,
        "target_table": targets,
        "color_flow_table": array("i", [t for f in flows for p in f for t in p]),
    }
    path = tmp_path / "flavor_switch_p0.json"
    write_switch_sidecar(path, catalogue, 17)
    expected = {"switch_lines": 3, "targets": pdgs}
    expected.update(color_flow_offset=17, color_flows=flows)
    assert path.read_text() == json.dumps(expected)


def test_switch_catalogue_rows_view_its_tables():
    reference = _switch_reference(TTJ_ROWS, [0.0, 0.0, 173.0, 173.0, 0.0])
    catalogue = switch_catalogue(reference, None, 2)
    table, rows = catalogue["ordering_table"], catalogue["orderings"]
    assert len(rows) == len(table.evaluation) == 7
    assert rows[-1] == rows[6] and rows[1:3] == [rows[1], rows[2]]
    assert [o["target"] for o in rows] == list(table.target)
    with pytest.raises(IndexError):
        rows[7]
    targets = catalogue["target_table"]
    assert catalogue["target_pdgs"] == [
        list(targets.pdgs[k * 5:(k + 1) * 5]) for k in range(len(targets.evaluation))
    ]


def _slot_pdgs(reference, evaluation):
    pdgs = reference["runtime_processes"][evaluation["process"]]["pdgs"]
    return tuple(pdg for _, pdg in sorted(zip(evaluation["legs"], pdgs, strict=True)))


def _conjugate(pdgs):
    return tuple(p if p == 21 else -p for p in pdgs)


@pytest.mark.parametrize("power", [2, 1, None])
def test_switch_shares_charge_conjugate_totals_of_pure_qcd(monkeypatch, power):
    import pyamplicol.artifacts.umami_reweight as module

    rows = ([21, 21, 21, 21], [2, -2, 21, 21], [2, -2, 1, -1], [1, 2, 1, 2])
    reference = _switch_reference(rows, [0.0] * 4)
    for process in reference["runtime_processes"]:
        process["alpha_s_power"] = power
    monkeypatch.setattr(module, "SWITCH_C_DEDUP", False)
    old = switch_catalogue(reference, None, 2)
    monkeypatch.setattr(module, "SWITCH_C_DEDUP", True)
    new = switch_catalogue(reference, None, 2)
    if power != 2:
        # not proven pure QCD (an electroweak coupling or no proof): unchanged
        assert new["calls"] == old["calls"]
        assert new["ordering_table"] == old["ordering_table"]
        return
    assert len(new["calls"]) < len(old["calls"])
    # every evaluation stays for the targets, which keep their own
    assert new["evaluations"] == old["evaluations"]
    assert list(new["targets"]) == list(old["targets"])
    assert list(new["target_pdgs"]) == list(old["target_pdgs"])
    written = {e for call in new["calls"] for e in call["evaluations"]}
    assert written == set(new["ordering_table"].evaluation)
    shared = 0
    for before, after in zip(old["orderings"], new["orderings"], strict=True):
        assert {**before, "evaluation": 0} == {**after, "evaluation": 0}
        own = _slot_pdgs(reference, old["evaluations"][before["evaluation"]])
        read = _slot_pdgs(reference, new["evaluations"][after["evaluation"]])
        # the total of the flavour or of its charge conjugate in the same slots
        assert read in (own, _conjugate(own))
        shared += read != own
    assert shared


@pytest.mark.parametrize("width", [None, {"lhablock": "DECAY", "lhacode": [6]},
                                   {"name": "particle.6.width"}])
def test_charge_conjugate_totals_cross_arrangements_without_a_width(
    monkeypatch, width
):
    import pyamplicol.artifacts.umami_reweight as module

    # t and t~ swap slots under C: the pairs join different arrangements
    rows = ([21, 21, 6, -6, 21, 21], [21, 21, -6, 6, 21, 21],
            [2, -2, 6, -6, 21, 21], [2, -2, -6, 6, 21, 21],
            [21, 21, 6, -6, 2, -2], [21, 21, -6, 6, 2, -2])
    reference = _switch_reference(rows, [0.0, 0.0, 173.0, 173.0, 0.0, 0.0])
    for process in reference["runtime_processes"]:
        process["alpha_s_power"] = 4
        process["parameters"] = [{"name": "MT"}]
        if width is not None:
            process["parameters"].append({"name": "WT", **width})
    monkeypatch.setattr(module, "SWITCH_C_DEDUP", False)
    old = switch_catalogue(reference, None, 2)
    monkeypatch.setattr(module, "SWITCH_C_DEDUP", True)
    new = switch_catalogue(reference, None, 2)
    if width is not None:
        # a top width breaks the symmetry of the leading colour: unchanged
        assert new["calls"] == old["calls"]
        assert new["ordering_table"] == old["ordering_table"]
        return
    assert len(new["calls"]) < len(old["calls"])
    arrangements: dict[int, set[int]] = {}
    for row in old["orderings"]:
        arrangements.setdefault(row["evaluation"], set()).add(row["arrangement"])
    crossing = 0
    for before, after in zip(old["orderings"], new["orderings"], strict=True):
        own = _slot_pdgs(reference, old["evaluations"][before["evaluation"]])
        read = _slot_pdgs(reference, new["evaluations"][after["evaluation"]])
        assert read in (own, _conjugate(own))
        crossing += before["arrangement"] not in arrangements[after["evaluation"]]
    assert crossing
