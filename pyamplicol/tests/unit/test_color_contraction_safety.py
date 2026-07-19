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
from pyamplicol.generation.dag_color import ColorEngine
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_ordering import _sector_intermediate_order_words
from pyamplicol.models.builtin.model import BuiltinSMModel
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

    assert set(sector) == {
        "id",
        "kind",
        "open_color_lines",
        "trace_labels",
        "singlet_labels",
        "word_labels",
        "coloured_label_groups",
        "line_label_groups",
        "color_words",
        "admissible_traversal_words",
    }
    assert "idenso_required" not in payload

    assert "quark_lines" not in sector
    assert line == {
        "fundamental_label": 2,
        "antifundamental_label": 1,
        "adjoint_labels": [4],
        "singlet_labels": [],
        "line_labels": [2, 4, 1],
    }
    assert not {"quark_label", "antiquark_label", "gluon_labels"} & line.keys()


def test_three_open_lines_keep_distinct_fixed_sink_traversals() -> None:
    plan = build_color_plan(build_process_ir("d d~ > u u~ s s~"))
    sector = plan.sectors[0]

    assert sector.color_words == ((2, 1, 3, 4, 5, 6),)
    assert _sector_intermediate_order_words(sector) == (
        (2, 1, 3, 4, 5, 6),
        (3, 4, 2, 1, 5, 6),
    )


def test_three_open_line_nlc_keeps_exact_qualified_coefficients() -> None:
    plan = build_color_plan(
        build_process_ir(
            "d d~ > u u~ s s~ g",
            color_accuracy="full",
        ),
        color_accuracy="full",
    )
    sectors = {sector.word_labels: sector for sector in plan.sectors}
    reference = sectors[(2, 1, 3, 4, 5, 7, 6)]

    assert color_contraction_factors(plan, reference, reference) == (
        81.0,
        72.0,
        72.0,
    )
    assert color_contraction_factors(
        plan,
        reference,
        sectors[(2, 1, 3, 6, 5, 7, 4)],
    ) == (0.0, -24.0, -24.0)
    assert color_contraction_factors(
        plan,
        reference,
        sectors[(2, 4, 3, 6, 5, 7, 1)],
    ) == (0.0, 8.0, 8.0)


def test_nlc_one_open_line_recycles_orderings_in_one_shared_dag() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > t t~ g", color_accuracy="nlc")
    plan = build_color_plan(process, color_accuracy="nlc")
    engine = ColorEngine(plan, model)
    dag = compile_generic_dag(process, model=model)

    assert plan.sector_count == 6
    assert engine.shared_lc_orderings is True
    assert len(dag.currents) == 250
    assert len(dag.interactions) == 624
    assert dag.interaction_evaluation_count == 348
    assert len(dag.amplitude_roots) == 192


def test_nlc_multiple_open_lines_keep_pairing_identity_sector_local() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > u u~ s s~", color_accuracy="nlc")
    plan = build_color_plan(process, color_accuracy="nlc")
    engine = ColorEngine(plan, model)

    assert plan.sector_count > 1
    assert engine.shared_lc_orderings is False


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
