# SPDX-License-Identifier: 0BSD
"""Crafted UMAMI metadata must not reach the generated reweight sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyamplicol.artifacts.umami_reweight import (
    load_umami_metadata,
    reweight_header,
    switch_catalogue,
    write_reweight_provider,
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


def test_valid_metadata_renders():
    header = reweight_header(_document(), [_document()], "lc", "full")
    assert "reweight_c0_f0_permutation[] = {0, 1, 2, 3};" in header


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
        reweight_header(document, [_document()], "lc", "full")


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


def test_switch_sums_the_arrangements_of_massive_quarks():
    # a conjugated flavour has t~ t in the slots of t t~
    rows = (*TTJ_ROWS, [-2, 2, -6, 6, 21])
    reference = _switch_reference(rows, [0.0, 0.0, 173.0, 173.0, 0.0])
    catalogue = switch_catalogue(reference, None, 2)
    assert _moved(catalogue, reference, {6}) == []
    assert {tuple(t[2:4]) for t in catalogue["target_pdgs"]} == {(6, -6), (-6, 6)}
    totals = {}
    for ordering in catalogue["orderings"]:
        pdf = tuple(ordering["pdf"])
        totals[pdf] = totals.get(pdf, 0.0) + ordering["weight"]
    assert totals == pytest.approx(dict.fromkeys(totals, 1.0))


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
