# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

import pyamplicol.color as color
from pyamplicol.color import (
    ColorGroupDescriptor,
    build_color_contraction_plan,
    build_color_plan,
    color_contraction_factors,
)
from pyamplicol.models.builtin.process_ir import build_process_ir


def test_color_one_open_line_factors_match_reference_convention() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > z g", color_accuracy="full"),
        color_accuracy="full",
    )
    sector = plan.sectors[0]
    assert color_contraction_factors(plan, sector, sector) == (9.0, 8.0, 8.0)


def test_color_plan_json_exposes_structural_open_line_roles() -> None:
    plan = build_color_plan(build_process_ir("d d~ > z g"))
    sector_record = plan.sectors[0]

    assert isinstance(sector_record.open_color_lines[0], color.LCOpenColorLine)
    assert not hasattr(sector_record, "quark_lines")
    assert not hasattr(color, "LCQuarkLine")

    payload = plan.to_json_dict()
    sector = payload["sectors"][0]
    line = sector["open_color_lines"][0]

    assert "quark_lines" not in sector
    assert line == {
        "fundamental_label": 2,
        "antifundamental_label": 1,
        "adjoint_labels": [4],
        "singlet_labels": [],
        "line_labels": [2, 4, 1],
    }
    assert not {"quark_label", "antiquark_label", "gluon_labels"} & line.keys()


def test_color_contraction_rejects_inconsistent_helicity_weights() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > z g", color_accuracy="full"),
        color_accuracy="full",
    )
    sector = plan.sectors[0]
    common = {
        "helicity_key": ("h:-1,+1,+0,+1",),
        "sector_id": sector.id,
        "word": tuple(sector.word_labels or sector.color_words[0]),
    }
    groups = (
        ColorGroupDescriptor(group_id=0, helicity_weight=1.0, **common),
        ColorGroupDescriptor(group_id=1, helicity_weight=0.5, **common),
    )

    with pytest.raises(ValueError, match="inconsistent helicity weights"):
        build_color_contraction_plan(plan, groups)


def test_color_contraction_public_names_have_no_legacy_aliases() -> None:
    assert callable(color.color_contraction_factor)
    assert callable(color.color_contraction_factors)
    assert not hasattr(color, "amplicol_color_factor")
    assert not hasattr(color, "amplicol_color_factors")
