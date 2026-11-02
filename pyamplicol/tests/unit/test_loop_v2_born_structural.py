# SPDX-License-Identifier: 0BSD
"""Lighter 2->6 generation, round 2 (M-FINAL mf-8g-1c), against its oracles.

* The structural loop Born DAG (``blocks.loop_born_dag`` with
  ``blocks.structural_helicities``: one helicity per leg whose source states
  differ in their helicity only) has the complete DAG's colour plan, the same
  (colour sector, coupling orders) of its roots and the same propagated
  species (``blocks.STRUCTURAL_BORN = False`` compiles the complete DAG, the
  oracle); slices generated either way export the same plan and native tree
  program, byte for byte, and close the same Born terms.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

FIXED = {"MB": "0"}

# name: (process, Born orders, particle veto, legs the structural Born fixes)
BORNS = {
    "gg_ggg": ("g g > g g g", {"QCD": 3, "QED": 0}, (), {1, 2, 3, 4, 5}),
    "uux_ddxg_ew": ("u u~ > d d~ g", {"QCD": 3, "QED": 2}, (), {5}),
    "uux_ggg": ("u u~ > g g g", {"QCD": 3, "QED": 0}, (), {3, 4, 5}),
    "gg_ttxg": ("g g > t t~ g", {"QCD": 3, "QED": 0}, (), {1, 2, 3, 4, 5}),
    "udx_wpg": ("u d~ > w+ g", {"QCD": 1, "QED": 1}, (), {3, 4}),
    "uux_emepg": ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}, (), {5}),
    "uux_ddxgg": ("u u~ > d d~ g g", {"QCD": 4, "QED": 0}, (), {5, 6}),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, (), {1, 2, 3, 4, 5, 6}),
    "gg_ggggg": ("g g > g g g g g", {"QCD": 5, "QED": 0}, (6,), set(range(1, 8))),
    "uux_ggggg": ("u u~ > g g g g g", {"QCD": 5, "QED": 0}, (), {3, 4, 5, 6, 7}),
}

# name: (process, Born orders, accuracy, particle veto)
SLICES = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, "full", ()),
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "full", ()),
    "uux_ggg": ("u u~ > g g g", {"QCD": 3, "QED": 0}, "full", ()),
    "gg_ggg_lc": ("g g > g g g", {"QCD": 3, "QED": 0}, "lc", ()),
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, "full", ()),
    "uux_emepg": ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}, "full", ()),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, "full", ()),
    "uux_ddxgg_lc": ("u u~ > d d~ g g", {"QCD": 4, "QED": 0}, "lc", ()),
}


@contextmanager
def _structural(on: bool):
    from pyamplicol.loops.v2 import blocks

    saved = blocks.STRUCTURAL_BORN
    blocks.STRUCTURAL_BORN = on
    try:
        yield
    finally:
        blocks.STRUCTURAL_BORN = saved


def _born_summary(compiled, name: str):
    """(selected helicities, colour plan, {sector: orders}, species) of the
    loop Born DAG of ``BORNS[name]``."""

    from pyamplicol.loops.v2 import blocks
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    process, born, veto, _fixed = BORNS[name]
    model = CompiledUFOModel(compiled)
    ir = build_model_process_ir(process, compiled.ir, color_accuracy="full")
    dag = blocks.loop_born_dag(
        ir,
        model,
        max_coupling_orders=born,
        ignored_particle_ids=blocks.vetoed_particle_ids(veto),
    )
    sectors: dict[int | None, set] = {}
    for root in blocks._born_roots(dag, model):
        sectors.setdefault(root.color_sector_id, set()).add(root.coupling_orders)
    return (
        blocks.structural_helicities(ir, model),
        dag.color_plan,
        sectors,
        blocks.propagated_species(dag),
        len(dag.amplitude_roots),
    )


def _assert_structural_born_is_the_complete_borns(compiled, name: str):
    with _structural(False):
        selected, plan, sectors, species, roots = _born_summary(compiled, name)
    assert selected == {}
    found, found_plan, found_sectors, found_species, found_roots = _born_summary(
        compiled, name
    )
    assert set(found) == BORNS[name][3]
    assert found_plan == plan
    assert found_sectors == sectors
    assert found_species == species
    assert 0 < found_roots < roots


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name",
    ["gg_ggg", "uux_ddxg_ew", "uux_ggg", "gg_ttxg", "udx_wpg", "uux_emepg", "uux_ddxgg"],
)
def test_structural_born_keeps_sectors_orders_and_species(loop_sm_compiled_model, name):
    """Gluons, a massive top and a W are fixed to one helicity, massless
    quarks and leptons keep both; the colour plan, every sector's root orders
    (mixed QCD/QED orders and the U(1)-carrying quark sectors included) and
    the propagated species are the complete DAG's."""

    from pyamplicol.models.loading import load_compiled_model

    _assert_structural_born_is_the_complete_borns(
        load_compiled_model(loop_sm_compiled_model), name
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_gggg", "gg_ggggg", "uux_ggggg"])
def test_structural_born_keeps_sectors_orders_and_species_at_six_and_seven_legs(
    loop_sm_compiled_model, name
):
    from pyamplicol.models.loading import load_compiled_model

    _assert_structural_born_is_the_complete_borns(
        load_compiled_model(loop_sm_compiled_model), name
    )


@pytest.fixture(scope="module")
def generate(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    # The committed QCD colour continuation (``test_loop_v2_native_trees``).
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )

    def run(name: str) -> LoopSlice:
        process, born, accuracy, veto = SLICES[name]
        loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
        request = SliceRequest(
            process,
            born,
            loop,
            accuracy=accuracy,
            fixed_parameters=FIXED,
            particle_veto=veto,
            colour_declaration=None if accuracy == "full" else declaration,
        )
        return LoopSlice.generate(compiled, request, loop_native)

    run.compiled = compiled
    return run


def _exported(slice_: LoopSlice) -> str:
    from pyamplicol.loops.v2.plan_export import export_plan

    exported = export_plan(slice_, forward=False, trees=True)
    return hashlib.sha256(
        (exported.json() + exported.trees.json()).encode()
    ).hexdigest()


def _assert_structural_slice_is_the_complete_borns(generate, name: str):
    from pyamplicol.loops.v2.colour import born_basis

    with _structural(False):
        oracle = generate(name)
    found = generate(name)
    assert found.blocks.born_dag.species == oracle.blocks.born_dag.species
    assert found.blocks.born_orders == oracle.blocks.born_orders
    assert born_basis(found.blocks) == born_basis(oracle.blocks)
    assert {k: len(v) for k, v in found.born_terms.items()} == {
        k: len(v) for k, v in oracle.born_terms.items()
    }
    assert _exported(found) == _exported(oracle)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name", ["gg_gg", "uux_ddxg", "uux_ggg", "gg_ggg_lc", "gg_ttx", "uux_emepg"]
)
def test_structural_born_exports_the_complete_borns_plan(generate, name):
    """The exported plan and native tree program are byte-identical."""

    _assert_structural_slice_is_the_complete_borns(generate, name)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_gggg", "uux_ddxgg_lc"])
def test_structural_born_exports_the_complete_borns_plan_at_six_legs(generate, name):
    _assert_structural_slice_is_the_complete_borns(generate, name)


@pytest.mark.loop_physics
def test_the_summary_recompiles_its_structural_and_its_complete_born(generate):
    """``BornDagSummary.structural`` is the DAG its roots index;
    ``materialize`` the complete Born DAG, every helicity included."""

    from pyamplicol.loops.v2 import blocks

    slice_ = generate("uux_ddxg")
    summary = slice_.blocks.born_dag
    assert isinstance(summary, blocks.BornDagSummary)
    structural = summary.structural()
    assert summary.roots == blocks._born_roots(structural, summary.model)
    complete = summary.materialize()
    assert len(complete.amplitude_roots) > len(structural.amplitude_roots)
    assert complete.color_plan == summary.color_plan
    assert blocks.propagated_species(complete) == summary.species
    assert {(r.color_sector_id, r.coupling_orders) for r in summary.roots} == {
        (r.color_sector_id, r.coupling_orders)
        for r in blocks._born_roots(complete, summary.model)
    }
