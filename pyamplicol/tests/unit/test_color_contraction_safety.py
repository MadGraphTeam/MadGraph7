# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from itertools import permutations
from math import isclose, nextafter

import pytest

import pyamplicol.color as color
from pyamplicol.color import (
    ColorContractionEntry,
    ColorGroupDescriptor,
    build_color_contraction_plan,
    build_color_plan,
    color_contraction_factors,
)
from pyamplicol.color.contraction_factors import (
    _pure_adjoint_full_factor_by_relative_permutation,
    _pure_adjoint_full_factor_uncached,
    _relative_adjoint_permutation,
)
from pyamplicol.generation.artifact_writer import _color_contraction
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


def test_pure_adjoint_full_factors_cache_relative_permutations_exactly() -> None:
    words = tuple(permutations((11, 13, 17, 19)))
    _pure_adjoint_full_factor_by_relative_permutation.cache_clear()

    for left in words:
        for right in words:
            relative = _relative_adjoint_permutation(left, right)
            assert relative is not None
            cached = _pure_adjoint_full_factor_by_relative_permutation(
                relative,
                len(left),
                20,
            )
            direct = _pure_adjoint_full_factor_uncached(
                left,
                right,
                len(left),
                20,
            )
            assert cached == direct

    info = _pure_adjoint_full_factor_by_relative_permutation.cache_info()
    assert info.misses == len(words)
    assert info.hits == len(words) * (len(words) - 1)


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


def test_color_contraction_compacts_identical_helicity_components() -> None:
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="full"),
        color_accuracy="full",
    )
    first, second = plan.sectors[:2]
    first_word = tuple(first.word_labels or first.color_words[0])
    second_word = tuple(second.word_labels or second.color_words[0])
    groups = (
        ColorGroupDescriptor(
            group_id=10,
            helicity_key=("helicity:a",),
            sector_id=second.id,
            word=second_word,
            helicity_weight=1.0,
        ),
        ColorGroupDescriptor(
            group_id=20,
            helicity_key=("helicity:b",),
            sector_id=first.id,
            word=first_word,
            helicity_weight=1.0,
        ),
        ColorGroupDescriptor(
            group_id=30,
            helicity_key=("helicity:a",),
            sector_id=first.id,
            word=first_word,
            helicity_weight=1.0,
        ),
        ColorGroupDescriptor(
            group_id=40,
            helicity_key=("helicity:b",),
            sector_id=second.id,
            word=second_word,
            helicity_weight=1.0,
        ),
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    assert contraction.entries == ()
    block = contraction.repeated_block
    assert block is not None
    assert block.component_count == 2
    assert block.component_group_ids == (30, 20, 10, 40)
    logical = tuple(contraction.iter_logical_entries())
    assert len(logical) == contraction.logical_entry_count
    assert tuple(
        (entry.left_group_id, entry.right_group_id) for entry in logical
    ) == tuple(
        (
            block.component_group_ids[
                entry.left_group_index * block.component_count + component
            ],
            block.component_group_ids[
                entry.right_group_index * block.component_count + component
            ],
        )
        for component in range(block.component_count)
        for entry in block.entries
    )
    payload = contraction.to_json_dict()
    assert payload["entry_count"] == 0
    assert payload["logical_entry_count"] == len(logical)
    assert payload["entries"] == []
    assert payload["repeated_block"] == block.to_json_dict()
    assert _color_contraction(payload)["repeated_block"] == block.to_json_dict()

    expanded = tuple(
        entry
        for helicity_key in (("helicity:a",), ("helicity:b",))
        for entry in (
            build_color_contraction_plan(
                plan,
                tuple(
                    descriptor
                    for descriptor in groups
                    if descriptor.helicity_key == helicity_key
                ),
            )
            or pytest.fail("single-component contraction is absent")
        ).entries
    )
    amplitudes = {
        10: complex(0.5, -1.0),
        20: complex(1.5, 0.25),
        30: complex(-0.75, 2.0),
        40: complex(0.125, -0.5),
    }

    def reduce(entries: tuple[ColorContractionEntry, ...]) -> float:
        total = 0.0
        for entry in entries:
            left = amplitudes[entry.left_group_id]
            right = amplitudes[entry.right_group_id]
            product = left * right.conjugate()
            total += entry.symmetry_factor * (
                entry.weight_re * product.real - entry.weight_im * product.imag
            )
        return total

    assert isclose(reduce(logical), reduce(expanded), rel_tol=1.0e-15, abs_tol=1.0e-15)


def test_full_color_permutation_orbit_emits_klein_four_walsh_plan() -> None:
    plan = build_color_plan(
        build_process_ir("d d~ > z g g g g", color_accuracy="full"),
        color_accuracy="full",
    )
    groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(plan.sectors)
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    block = contraction.repeated_block
    assert block is not None
    factorized = block.factorized_block
    assert factorized is not None
    assert factorized.kind == "klein-four-walsh"
    assert len(factorized.cosets) == len(plan.sectors) // 4
    assert sorted(index for coset in factorized.cosets for index in coset) == list(
        range(len(plan.sectors))
    )
    payload = contraction.to_json_dict()
    assert payload["repeated_block"] == block.to_json_dict()
    assert _color_contraction(payload)["repeated_block"] == block.to_json_dict()


def test_walsh_factorization_is_full_color_only_and_falls_back_safely() -> None:
    nlc_plan = build_color_plan(
        build_process_ir("d d~ > z g g g g", color_accuracy="nlc"),
        color_accuracy="nlc",
    )
    nlc_groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(nlc_plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=tuple(sector.word_labels or sector.color_words[0]),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(nlc_plan.sectors)
    )
    nlc_contraction = build_color_contraction_plan(nlc_plan, nlc_groups)
    assert nlc_contraction is not None
    assert nlc_contraction.repeated_block is not None
    assert nlc_contraction.repeated_block.factorized_block is None

    full_plan = build_color_plan(
        build_process_ir("d d~ > z g g g g", color_accuracy="full"),
        color_accuracy="full",
    )
    duplicate_word = tuple(
        full_plan.sectors[0].word_labels or full_plan.sectors[0].color_words[0]
    )
    malformed_groups = tuple(
        ColorGroupDescriptor(
            group_id=component_index * len(full_plan.sectors) + sector_index,
            helicity_key=(f"helicity:{component_index}",),
            sector_id=sector.id,
            word=(
                duplicate_word
                if sector_index == 1
                else tuple(sector.word_labels or sector.color_words[0])
            ),
            helicity_weight=1.0,
        )
        for component_index in range(2)
        for sector_index, sector in enumerate(full_plan.sectors)
    )
    malformed_contraction = build_color_contraction_plan(full_plan, malformed_groups)
    assert malformed_contraction is not None
    assert malformed_contraction.repeated_block is not None
    assert malformed_contraction.repeated_block.factorized_block is None


def test_color_contraction_compaction_falls_back_for_nonidentical_components() -> None:
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="full"),
        color_accuracy="full",
    )
    sector = plan.sectors[0]
    common = {
        "sector_id": sector.id,
        "word": tuple(sector.word_labels or sector.color_words[0]),
    }
    groups = (
        ColorGroupDescriptor(
            group_id=0,
            helicity_key=("helicity:a",),
            helicity_weight=1.0,
            **common,
        ),
        ColorGroupDescriptor(
            group_id=1,
            helicity_key=("helicity:b",),
            helicity_weight=nextafter(1.0, 2.0),
            **common,
        ),
    )

    contraction = build_color_contraction_plan(plan, groups)

    assert contraction is not None
    assert contraction.repeated_block is None
    assert contraction.logical_entry_count == len(contraction.entries) == 2


def test_color_contraction_public_names_have_no_legacy_aliases() -> None:
    assert callable(color.color_contraction_factor)
    assert callable(color.color_contraction_factors)
    assert not hasattr(color, "amplicol_color_factor")
    assert not hasattr(color, "amplicol_color_factors")
