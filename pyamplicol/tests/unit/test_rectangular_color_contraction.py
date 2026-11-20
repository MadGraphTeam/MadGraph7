# SPDX-License-Identifier: 0BSD
"""The array fast path of recurrence colour contraction equals the descriptor path."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from pyamplicol.color import build_color_plan
from pyamplicol.generation import recurrence_physics as physics
from pyamplicol.generation.recurrence_color import (
    encode_recurrence_color_contraction,
    recurrence_color_contraction_digest,
)
from pyamplicol.models.builtin.process_ir import build_process_ir

_HELICITIES = ((-1, -1, 1, 1, 1), (1, -1, -1, 1, 1), (1, 1, -1, -1, 1))


def _case(process: str, basis: str = "trace"):
    ir = build_process_ir(process, color_accuracy="full")
    plan = build_color_plan(ir, color_accuracy="full", basis=basis)
    labels = tuple(leg.label for leg in ir.legs)
    slot = labels.index
    logical = SimpleNamespace(
        layout="contracted-color-union",
        replay_partitions=(),
        external_legs=tuple(
            SimpleNamespace(source_slot=index, public_label=label)
            for index, label in enumerate(labels)
        ),
        physical_sectors=tuple(
            SimpleNamespace(
                sector_id=sector.id,
                kind=sector.kind,
                word_source_slots=tuple(slot(label) for label in sector.color_words[0]),
                open_strings=tuple(
                    SimpleNamespace(
                        fundamental_source_slot=slot(line.fundamental_label),
                        adjoint_source_slots=tuple(
                            slot(label) for label in line.adjoint_labels
                        ),
                        antifundamental_source_slot=slot(line.antifundamental_label),
                        singlet_source_slots=tuple(
                            slot(label) for label in line.singlet_labels
                        ),
                    )
                    for line in sector.open_color_lines
                ),
            )
            for sector in plan.sectors
        ),
    )
    return logical, plan


def _labels(logical):
    return {leg.source_slot: leg.public_label for leg in logical.external_legs}


def _owners(logical):
    sector_ids = {sector.sector_id for sector in logical.physical_sectors}
    return physics.recurrence_color_sector_owner_map(logical, sector_ids, set())


def _contract(
    logical, plan, helicities, destinations, zero=(), *, fast: bool, monkeypatch
):
    with monkeypatch.context() as patch:
        if not fast:
            patch.setattr(
                physics,
                "build_rectangular_symmetric_group_contraction_plan",
                lambda *_args, **_kwargs: None,
            )
        try:
            return physics.build_recurrence_color_contraction(
                logical, plan, helicities, destinations, zero, contraction="auto"
            )
        except ValueError as exc:
            return f"ValueError: {exc}"


def _payload(logical, plan, helicities, destinations, contraction) -> str:
    try:
        return recurrence_color_contraction_digest(
            _encode(logical, plan, helicities, destinations, contraction)
        )
    except ValueError as exc:  # shuffled helicities are not codec component order
        return f"ValueError: {exc}"


def _encode(logical, plan, helicities, destinations, contraction) -> bytes:
    coordinates = np.asarray(destinations, dtype=np.int64)[
        np.asarray(contraction.destination_by_group, dtype=np.int64)
    ]
    sectors = coordinates[:, 0]
    return encode_recurrence_color_contraction(
        contraction,
        sector_count=len(logical.physical_sectors),
        component_count=len(helicities),
        ordered_group_ids=contraction.symmetric_group_block.component_group_ids,
        destination_by_group=contraction.destination_by_group,
        destination_count=len(destinations),
        group_sector_ids=sectors,
        group_component_ids=coordinates[:, 1],
        sector_owner_ids=physics.recurrence_color_sector_owner_map(
            logical, set(sectors.tolist()), set()
        ),
        exact_coefficients=physics.recurrence_exact_color_coefficients(
            plan, contraction, sectors.tolist()
        ),
    )


@pytest.mark.parametrize(
    ("process", "basis"),
    (
        ("g g > g g g", "trace"),
        ("g g > g g g", "adjoint"),
        ("u g > u g g", "trace"),
        ("u u~ > u u~ g g", "trace"),
        ("u u~ > d d~ g g", "trace"),
    ),
)
@pytest.mark.parametrize("layout", ("dense", "owners", "shuffled"))
def test_fast_path_matches_descriptor_path(process, basis, layout, monkeypatch):
    logical, plan = _case(process, basis)
    owners = _owners(logical)
    sector_ids = [sector.sector_id for sector in logical.physical_sectors]
    if layout == "owners":
        sector_ids = [
            sector_id for sector_id in sector_ids if owners[sector_id] == sector_id
        ]
    destinations = [
        (sector_id, helicity) for sector_id in sector_ids for helicity in range(3)
    ]
    if layout == "shuffled":
        random.Random(7).shuffle(destinations)
    destinations = tuple(destinations)
    fast = physics._rectangular_recurrence_color_contraction(
        logical, plan, _HELICITIES, destinations, (), _labels(logical)
    )
    reference = _contract(
        logical, plan, _HELICITIES, destinations, fast=False, monkeypatch=monkeypatch
    )
    actual = _contract(
        logical, plan, _HELICITIES, destinations, fast=True, monkeypatch=monkeypatch
    )
    assert actual == reference
    if fast is not None:
        assert fast == reference
        assert _payload(logical, plan, _HELICITIES, destinations, fast) == _payload(
            logical, plan, _HELICITIES, destinations, reference
        )
    assert fast is not None


@pytest.mark.parametrize(
    "variation",
    (
        "duplicate-helicity",
        "missing-destination",
        "unknown-sector",
        "no-helicity",
        "zero",
    ),
)
def test_fast_path_defers_irregular_domains(variation, monkeypatch):
    logical, plan = _case("u u~ > u u~ g g")
    helicities = _HELICITIES
    owners = _owners(logical)
    destinations = [
        (sector.sector_id, helicity)
        for sector in logical.physical_sectors
        for helicity in range(3)
    ]
    zero: tuple[int, ...] = ()
    if variation == "duplicate-helicity":
        helicities = (*_HELICITIES[:2], _HELICITIES[0])
    elif variation == "missing-destination":
        destinations.pop(4)
    elif variation == "unknown-sector":
        destinations.append((len(logical.physical_sectors), 0))
    elif variation == "no-helicity":
        destinations[2] = (destinations[2][0], None)
    else:
        zero_owner = max(owners)
        zero = tuple(
            sector_id for sector_id, owner in enumerate(owners) if owner == zero_owner
        )
        destinations = [item for item in destinations if item[0] not in zero]
    destinations = tuple(destinations)
    fast = physics._rectangular_recurrence_color_contraction(
        logical, plan, helicities, destinations, zero, _labels(logical)
    )
    reference = _contract(
        logical,
        plan,
        helicities,
        destinations,
        zero,
        fast=False,
        monkeypatch=monkeypatch,
    )
    actual = _contract(
        logical,
        plan,
        helicities,
        destinations,
        zero,
        fast=True,
        monkeypatch=monkeypatch,
    )
    assert actual == reference
    if variation in {"missing-destination", "zero"}:
        # An absent alias or a certified-zero class can leave the domain rectangular.
        assert fast is None or fast == reference
    else:
        assert fast is None
