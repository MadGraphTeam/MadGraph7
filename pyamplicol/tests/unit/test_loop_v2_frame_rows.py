# SPDX-License-Identifier: 0BSD
"""The frame cost's per-leg helicity rows (genericity audit m4, package
native-seeds).

A segment's rows are the helicity assignments of its legs, so the
cheapest-frame policy prices it by the product of its legs' helicity counts
(``generator_input.leg_rows``: 1 for a scalar, 3 for a massive vector, ...).
The native policy with ``leg_rows`` chooses the frames of the Python
``cheapest(frame_rows(leg_rows), model)``, the argmin over every frame. The
counts are required: no count per leg is assumed (second genericity audit,
m8), so a call without them is refused.
"""

from __future__ import annotations

import json

import pytest

from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    ModelBlockTable,
    ModelLoopCatalogue,
    generate_families,
)
from pyamplicol.loops.v2.generator_input import NativeGeneration, frame_rows, leg_rows
from pyamplicol.loops.v2.schedules import STRUCTURED, build_schedules, cheapest
from pyamplicol.loops.v2.skeletons import DressedColour
from tests.unit.test_loop_v2_generate import assert_same_schedules


def _frames(plan) -> dict:
    return {key: schedule.trie.frame for key, schedule in plan.schedules.items()}


@pytest.mark.loop_fast
def test_frame_rows_is_the_product_of_the_legs_rows():
    for mask in range(1 << 6):
        assert frame_rows((2,) * 6)(mask) == 2 ** mask.bit_count()
    rows = frame_rows((1, 3, 2, 5))
    assert [rows(mask) for mask in (0, 0b1, 0b10, 0b110, 0b1111)] == [1, 1, 3, 6, 30]


@pytest.mark.loop_fast
def test_native_frames_with_leg_rows_equal_python(loop_native):
    """u u~ > d d~ g on the QCD+QED control model: two rows per leg (its
    massless legs) give the Python two-rows-per-leg plan; uneven counts
    change frames, and the native policy picks the Python policy's frames
    (equal skeletons, views and tries); no counts are refused."""

    from pyamplicol.loops.control_models import qcd_qed_control_model

    model = qcd_qed_control_model()
    catalogue = ModelLoopCatalogue(model)
    selection = LoopOrderSelection(frozenset({"QCD"}), frozenset({(("QCD", 5),)}))
    table = ModelBlockTable(model, ["u", "u~"], ["d", "d~", "g"], cap=selection.cap)
    families = generate_families(table, catalogue, selection)
    colour = DressedColour()

    def native(rows):
        generation = NativeGeneration(table, catalogue, selection, loop_native)
        return generation.schedules(colour, leg_rows=rows)

    with pytest.raises(ValueError, match="needs the helicity rows of every leg"):
        native(None)
    default = native((2,) * 5)
    assert_same_schedules(
        default,
        build_schedules(
            families,
            catalogue,
            colour,
            policy=cheapest(frame_rows((2,) * 5), STRUCTURED),
        ),
    )
    uneven = UNEVEN_ROWS
    plan = native(uneven)
    python = build_schedules(
        families, catalogue, colour, policy=cheapest(frame_rows(uneven), STRUCTURED)
    )
    assert_same_schedules(plan, python)
    changed = [k for k, f in _frames(plan).items() if f != _frames(default)[k]]
    assert changed  # the counts decide frames
    generation = NativeGeneration(table, catalogue, selection, loop_native)
    for wrong in ((2,) * 4, (2, 2, 0, 2, 2)):
        with pytest.raises(ValueError, match="leg_rows"):
            generation.schedules(colour, leg_rows=wrong)


# A row count per leg (u, u~, d, d~, g) that moves frames off the
# two-per-leg choice.
UNEVEN_ROWS = (1, 5, 1, 4, 3)


@pytest.mark.loop_fast
def test_leg_rows_of_a_massive_vector(loop_sm_compiled_model, loop_native):
    """u u~ > z g on loop_sm: the Z has three helicity rows, every other leg
    two; with them the native frames are the Python policy's."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.blocks import build_tree_blocks
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    born, bounds = {"QCD": 1, "QED": 1}, {"QCD": 3, "QED": 1}
    prepared = prepare_loop_request_model(
        compiled,
        born_order_bounds=born,
        loop_order_bounds=bounds,
        fixed_parameters={"MB": "0"},
    )
    data = json.loads(prepared.structure.native.to_json())
    catalogue = ModelLoopCatalogue(
        data, species_orders=prepared.structure.perturbative_species_orders
    )
    process = build_model_process_ir("u u~ > z g", compiled.ir, color_accuracy="full")
    blocks = build_tree_blocks(
        process,
        CompiledUFOModel(compiled),
        born_coupling_orders=born,
        block_coupling_orders=bounds,
    )
    rows = leg_rows(blocks)
    assert rows == (2, 2, 3, 2)
    table = blocks.block_table(data)
    selection = LoopOrderSelection(
        frozenset({"QCD"}), frozenset({tuple(sorted(bounds.items()))})
    )
    generation = NativeGeneration(table, catalogue, selection, loop_native)
    colour = DressedColour()
    python = build_schedules(
        generate_families(table, catalogue, selection),
        catalogue,
        colour,
        policy=cheapest(frame_rows(rows), STRUCTURED),
    )
    assert_same_schedules(generation.schedules(colour, leg_rows=rows), python)
