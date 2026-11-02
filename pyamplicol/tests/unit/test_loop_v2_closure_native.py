# SPDX-License-Identifier: 0BSD
"""Lighter 2->6 generation (M-FINAL mf-8g-1b) against its oracles.

* The closed root colours of the counterterm and Born closures
  (``ct_sites.SiteClosure``): natively in batches (``loop_root_colours``,
  ``NATIVE_ROOT_COLOURS``) and one ``colour.reduce_network`` each (the
  oracle) give the same root terms, Born terms and exported native tree
  program, byte for byte (four and five legs here, six and seven nightly).
* The loop Born DAG of a request is compiled once (``blocks.born_dag_scope``):
  the public route's preflight (``artifact.born_amplitude_orders``) and the
  slice share it, and the slice keeps its ``BornDagSummary`` (the DAG is
  released): its roots, colour plan and propagated species are the DAG's,
  and it recompiles an equal DAG on demand.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

FIXED = {"MB": "0"}

# name: (process, Born orders, accuracy, particle veto)
PROCESSES = {
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, "full", ()),
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "full", ()),
    "gg_ggg_lc": ("g g > g g g", {"QCD": 3, "QED": 0}, "lc", ()),
    "uux_ddxg_nlc": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, "nlc", ()),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, "full", ()),
    "gg_ggggg_lc": ("g g > g g g g g", {"QCD": 5, "QED": 0}, "lc", (6,)),
}


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
        process, born, accuracy, veto = PROCESSES[name]
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


@contextmanager
def _root_colours(native: bool):
    from pyamplicol.loops.v2 import ct_sites

    saved = ct_sites.NATIVE_ROOT_COLOURS
    ct_sites.NATIVE_ROOT_COLOURS = native
    try:
        yield
    finally:
        ct_sites.NATIVE_ROOT_COLOURS = saved


def _term_key(term) -> tuple:
    """A root or Born term by value: its colour, sign, blocks and kernel
    reading (the entry by its catalogue key: two generations hold distinct
    scalar programs)."""

    from pyamplicol.loops.v2.ct_sites import _entry_key

    blocks = tuple(block.id for block in term.blocks)
    if hasattr(term, "entry"):
        return (
            term.colour,
            term.sign,
            blocks,
            term.kernel_legs,
            term.binding,
            _entry_key(term.entry),
        )
    grades = tuple((degree, str(value)) for degree, value in term.grades)
    return (term.colour, term.sign, blocks, term.pairings, grades, term.parameters)


def _closure(slice_: LoopSlice):
    root = [
        (
            _term_key(t.term),
            t.root,
            t.component,
            t.power,
            t.orders,
            t.coupling,
            t.multiplicity,
            t.scale,
            t.grade,
        )
        for t in slice_.counterterm_program.closure.terms
    ]
    born = {
        order: [_term_key(term) for term in terms]
        for order, terms in slice_.born_terms.items()
    }
    return root, born


def _trees_digest(slice_: LoopSlice) -> str:
    from pyamplicol.loops.v2.plan_export import export_plan

    exported = export_plan(slice_, forward=False, trees=True)
    return hashlib.sha256(
        (exported.json() + exported.trees.json()).encode()
    ).hexdigest()


def _assert_native_colours_equal_the_oracle(generate, name: str, export: bool):
    with _root_colours(False):
        oracle = generate(name)
    native = generate(name)
    terms, born = _closure(native)
    oracle_terms, oracle_born = _closure(oracle)
    assert len(terms) == len(oracle_terms) > 0
    assert terms == oracle_terms
    assert born == oracle_born
    if export:
        assert _trees_digest(native) == _trees_digest(oracle)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg", "gg_ggg_lc", "uux_ddxg_nlc"])
def test_native_root_colours_equal_the_python_closure(generate, name):
    """Root terms (whole-rule continuations at LC/NLC included), Born terms
    and the exported plan and tree program are the oracle's."""

    from pyamplicol.loops.v2 import ct_sites

    assert ct_sites.NATIVE_ROOT_COLOURS is True
    _assert_native_colours_equal_the_oracle(generate, name, export=True)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_gggg", "gg_ggggg_lc"])
def test_native_root_colours_equal_the_python_closure_at_six_and_seven_legs(
    generate, name
):
    _assert_native_colours_equal_the_oracle(generate, name, export=False)


@pytest.mark.loop_physics
def test_the_born_dag_is_compiled_once_and_summarised(generate, monkeypatch):
    """The preflight and the slice share one compile; the slice's summary
    holds the DAG's roots, colour plan and species and recompiles it."""

    from pyamplicol.loops.v2 import blocks
    from pyamplicol.loops.v2.artifact import born_amplitude_orders
    from pyamplicol.models.external import CompiledUFOModel

    compiles = []
    original = blocks.compile_generic_dag

    def counted(*args, **kwargs):
        compiles.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(blocks, "compile_generic_dag", counted)
    process, born, _accuracy, veto = PROCESSES["uux_ddxg"]
    compiled = generate.compiled
    with blocks.born_dag_scope():
        orders, dag = born_amplitude_orders(compiled, process, born, veto)
        slice_ = generate("uux_ddxg")
    assert len(compiles) == 1
    summary = slice_.blocks.born_dag
    assert isinstance(summary, blocks.BornDagSummary)
    assert summary.color_plan == dag.color_plan
    assert summary.species == blocks.propagated_species(dag)
    assert {root.coupling_orders for root in summary.roots} == set(orders)
    assert summary.roots == blocks._born_roots(dag, CompiledUFOModel(compiled))
    # Recompiled on demand (``materialize`` and attribute reads), equal.
    rebuilt = summary.materialize()
    assert len(compiles) == 2
    assert rebuilt.amplitude_roots == dag.amplitude_roots
    assert rebuilt.currents == dag.currents
    assert summary.interactions == dag.interactions
    # Without a scope, a slice compiles its own Born once.
    compiles.clear()
    generate("uux_ddxg")
    assert len(compiles) == 1
