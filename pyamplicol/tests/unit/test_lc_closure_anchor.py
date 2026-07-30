# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace

from pyamplicol.color import build_color_plan
from pyamplicol.generation.dag_algorithms import (
    infer_minimal_coupling_order_limits,
    prune_dag_to_amplitude_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _sector_for_word(expression: str, word: tuple[int, ...]):
    process = build_process_ir(expression)
    plan = build_color_plan(process)
    return process, next(
        sector for sector in plan.sectors if sector.word_labels == word
    )


def test_two_line_closure_rotates_public_blocks_from_source_slot_zero() -> None:
    process, sector = _sector_for_word("d d~ > t t~ g g", (2, 5, 6, 4, 3, 1))

    assert sector.canonical_closure_traversal_word(process) == (3, 1, 2, 5, 6, 4)
    assert sector.canonical_closure_sink_label(process) == 4
    assert sector.word_labels == (2, 5, 6, 4, 3, 1)


def test_public_reference_selects_sector_without_overriding_private_closure() -> None:
    process, sector = _sector_for_word(
        "d d~ > t t~ g g",
        (2, 5, 6, 4, 3, 1),
    )
    model = BuiltinSMModel()
    plan = build_color_plan(process)
    limits = infer_minimal_coupling_order_limits(
        process,
        model=model,
        selected_color_sector_ids=(sector.id,),
    )
    dag = prune_dag_to_amplitude_roots(
        compile_generic_dag(
            process,
            model=model,
            color_plan=plan,
            reference_color_order=sector.word_labels,
            selected_color_sector_ids=(sector.id,),
            max_coupling_orders=limits,
            closure_side_mask_pruning=False,
            color_order_mask_pruning=False,
            species_reachability_pruning=False,
        )
    )

    assert plan.sectors[sector.id].word_labels == (2, 5, 6, 4, 3, 1)
    assert (
        len(dag.currents),
        len(dag.interactions),
        len(dag.amplitude_roots),
        sum(
            current.index.external_mask.bit_count() == 1
            for current in dag.currents
        ),
    ) == (78, 150, 32, 12)


def test_closure_reconstructs_blocks_independently_of_line_tuple_order() -> None:
    process, sector = _sector_for_word("d d~ > t t~ g", (2, 5, 4, 3, 1))
    reordered = replace(
        sector,
        open_color_lines=tuple(reversed(sector.open_color_lines)),
    )

    assert reordered.canonical_closure_traversal_word(process) == (3, 1, 2, 5, 4)
    assert reordered.canonical_closure_sink_label(process) == 4


def test_closure_uses_source_position_not_numeric_public_label() -> None:
    process, sector = _sector_for_word("d d~ > t t~", (2, 4, 3, 1))
    label_map = {1: 10, 2: 5, 3: 20, 4: 30}
    relabelled_process = replace(
        process,
        legs=tuple(
            replace(leg, label=label_map[leg.label])
            for leg in process.legs
        ),
    )
    relabelled_sector = replace(
        sector,
        word_labels=tuple(label_map[label] for label in sector.word_labels),
        open_color_lines=tuple(
            replace(
                line,
                fundamental_label=label_map[line.fundamental_label],
                antifundamental_label=label_map[line.antifundamental_label],
                adjoint_labels=tuple(
                    label_map[label] for label in line.adjoint_labels
                ),
            )
            for line in sector.open_color_lines
        ),
    )

    assert relabelled_process.legs[0].label == 10
    assert min(label_map.values()) == 5
    assert relabelled_sector.canonical_closure_traversal_word(
        relabelled_process
    ) == (20, 10, 5, 30)


def test_singlet_source_zero_selects_minimum_coloured_source_slot() -> None:
    process = build_process_ir("e- d > e- d t t~")
    plan = build_color_plan(process)
    sector = next(
        sector
        for sector in plan.sectors
        if sector.word_labels == (4, 6, 5, 2)
    )

    assert process.legs[0].is_singlet
    assert process.legs[1].label == 2
    assert sector.canonical_closure_traversal_word(process) == (5, 2, 4, 6)
    assert sector.canonical_closure_sink_label(process) == 6


def test_single_trace_and_singlet_sectors_keep_public_words() -> None:
    for expression in ("g g > g g", "z z > z z"):
        process = build_process_ir(expression)
        plan = build_color_plan(process)
        assert all(
            sector.canonical_closure_traversal_word(process)
            == sector.color_words[0]
            for sector in plan.sectors
        )
