# SPDX-License-Identifier: 0BSD
"""Native loop-request DAG builder against the tree engine (WP3.1 A1-A2).

The oracle is ``GenericDAGCompiler(loop_blocks=True)`` with
``loops.v2.blocks._blocks``: the native DAG must decode to an equal
``GenericDAG`` (same currents, sources and interactions in the same order),
have the same flat-record digest (every float bit), and group into equal
``Block`` records for each tree-kernel variant. ``RUST-MIGRATION.md`` §3.A.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.generation.dag_compiler import GenericDAGCompiler
from pyamplicol.loops.v2 import dag_input
from pyamplicol.loops.v2.blocks import (
    RequestedLoopBlocks,
    _blocks,
    ward_breaking_fields,
)

VARIANTS = ("transverse", "full")


@pytest.fixture(scope="module")
def models(loop_sm_compiled_model):
    """(compiled loop_sm, tree-engine model)."""

    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    return compiled, CompiledUFOModel(compiled)


def _process(compiled, text):
    from pyamplicol.processes.model import build_model_process_ir

    return build_model_process_ir(text, compiled.ir, color_accuracy="full")


def _python(model, process, orders, **options):
    return GenericDAGCompiler(
        model=model, max_coupling_orders=orders, loop_blocks=True, **options
    ).compile(process)


def _native(model, process, orders, **options):
    reachability = options.pop("loop_block_reachability", None)
    return dag_input.native_blocks(
        process,
        model,
        max_coupling_orders=orders,
        reachability=reachability,
        **options,
    )


def _propagated(dag, model):
    """The current scan of ``ward_breaking_fields`` before its width test."""

    sources = set(dag.sources)
    return frozenset(
        int(current.index.particle_id)
        for current_id, current in enumerate(dag.currents)
        if current_id not in sources
        and current.index.auxiliary_kind is None
        and abs(int(model.color_rep(current.index.particle_id))) != 1
    )


def _assert_equal(compiled, model, text, orders, **options):
    process = _process(compiled, text)
    python = _python(model, process, orders, **options)
    native = _native(model, process, orders, **dict(options))
    before = dag_input.decode_count()
    assert native.decode() == python
    assert dag_input.decode_count() == before + 1
    assert native.counts() == (
        len(python.currents),
        len(python.interactions),
        len(python.sources),
    )
    # Bit equality of every field, floats included.
    assert native.digest() == dag_input.NativeDag.from_dag(python, model).digest()
    for variant in VARIANTS:
        expected = _blocks(python, model, variant)
        assert native.blocks(variant) == expected
    assert native.coloured_propagated() == _propagated(python, model)
    return python, native


CONTROLS = (
    ("g g > g g", {"QCD": 4}),
    ("u u~ > d d~ g", {"QCD": 5}),
    ("g g > g g g", {"QCD": 5}),
)


@pytest.mark.loop_fast
@pytest.mark.parametrize(("text", "orders"), CONTROLS)
def test_native_loop_blocks_equal_the_tree_engine(models, text, orders):
    compiled, model = models
    python, _native_dag = _assert_equal(compiled, model, text, orders)
    assert python.interactions


@pytest.mark.loop_fast
def test_python_dag_round_trips_through_the_columns(models):
    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    python = _python(model, process, {"QCD": 5})
    data = dag_input.encode_dag(python)
    plan = python.color_plan
    assert dag_input._decode(data, process, plan) == python
    handle = dag_input.NativeDag.from_dag(python, model)
    assert handle.columns() == data
    assert handle.decode() == python
    assert handle.blocks("transverse") == _blocks(python, model, "transverse")


@pytest.mark.loop_fast
def test_native_build_is_deterministic(models):
    compiled, model = models
    process = _process(compiled, "g g > g g g")
    first = _native(model, process, {"QCD": 5})
    second = _native(model, process, {"QCD": 5})
    assert first.columns() == second.columns()


@pytest.mark.loop_fast
def test_native_builder_refuses_what_it_does_not_cover(models):
    compiled, model = models
    with pytest.raises(dag_input.NativeDagUnsupported, match="model type"):
        dag_input.dag_input(_process(compiled, "g g > g g"), object())
    with pytest.raises(dag_input.NativeDagUnsupported, match="coloured external"):
        dag_input.dag_input(
            _process(compiled, "e+ e- > e+ e-"), model, max_coupling_orders={}
        )
    with pytest.raises(ValueError, match="full-colour"):
        from pyamplicol.processes.model import build_model_process_ir

        dag_input.dag_input(
            build_model_process_ir("g g > g g", compiled.ir, color_accuracy="lc"),
            model,
        )


def test_native_builder_sources_never_name_symbolica():
    """Rule 4: the builder and its binding make no Symbolica (licence) call.

    The exact rationals of the reuse derivation (``dag/exact.rs``) are the one
    use: Symbolica's ``Rational`` (a ``numerica`` domain, no licence check)."""

    root = Path(__file__).resolve().parents[2] / "rust"
    sources = [
        *sorted((root / "crates/rusticol-core/src/loop_v2/dag").glob("*.rs")),
        root / "loop-reduction/python/src/v2_dag.rs",
    ]
    assert len(sources) >= 8
    for path in sources:
        text = path.read_text()
        if path.name == "exact.rs":
            assert text.count("symbolica::") == 1, path
            assert "use symbolica::domains::rational::Rational;" in text
        else:
            assert "symbolica::" not in text, path


@pytest.mark.loop_nightly
def test_native_builder_needs_no_symbolica_licence(models, tmp_path):
    """The Rust builder runs on its tables alone, in a process without a key.

    The child imports the extension, which starts a restricted Symbolica
    instance: like ``test_six_gluon_generation_completes_without_a_personal_key``
    it must not overlap another unlicensed Symbolica process of the user.
    """

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    tables = dag_input.dag_input(process, model, max_coupling_orders={"QCD": 5})
    expected = _native(model, process, {"QCD": 5}).digest()
    path = tmp_path / "tables.json"
    path.write_text(tables)
    environment = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    script = (
        "import sys\n"
        "from pyamplicol.loops.native import load_loop_native\n"
        "dag = load_loop_native().V2Dag.build(open(sys.argv[1]).read())\n"
        "print(dag.digest())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip().splitlines()[-1] == expected


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "orders"),
    (
        ("u u~ > g g g", {"QCD": 5}),
        ("u u~ > e- e+ g g", {"QCD": 4, "QED": 2}),
        ("e+ e- > u u~ g", {"QCD": 3, "QED": 2}),
        ("u d~ > e+ ve g", {"QCD": 3, "QED": 2}),
        ("g g > t t~", {"QCD": 4}),
    ),
)
def test_native_loop_blocks_wider_controls(models, text, orders):
    compiled, model = models
    _assert_equal(compiled, model, text, orders)


@pytest.mark.loop_physics
def test_native_loop_blocks_with_top_widths_select_full_kernels(models):
    """g g > t t~ g g propagates tops with W_T != 0: the Ward query names
    them, and the full-kernel blocks are equal too (coordinator note 5)."""

    compiled, model = models
    python, native = _assert_equal(compiled, model, "g g > t t~ g g", {"QCD": 6})
    top = next(r for r in compiled.ir.particles if r.name == "t")
    assert int(top.pdg_code) in native.coloured_propagated()
    assert {"t", "t~"} <= ward_breaking_fields((python,), compiled)


@pytest.mark.loop_physics
def test_native_loop_blocks_reachability_and_ignored_ids(models):
    from pyamplicol.loops.v2.ct_closure import RootReachability

    compiled, model = models
    _assert_equal(
        compiled, model, "g g > g g g", {"QCD": 5}, ignored_particle_ids=(6, -6)
    )
    _assert_equal(compiled, model, "g g > g g", {"QCD": 4}, ignored_particle_ids=(21,))
    _assert_equal(
        compiled, model, "u u~ > d d~ g", {"QCD": 5}, ignored_vertex_kinds=(1, 2, 3)
    )
    process = _process(compiled, "g g > g g g")
    full = (1 << len(process.legs)) - 1
    _assert_equal(
        compiled,
        model,
        "g g > g g g",
        {"QCD": 5},
        loop_block_reachability=RootReachability(full),
    )
    python = _python(model, process, {"QCD": 5})
    rows = sorted(
        {
            (c.index.external_mask, c.index.particle_id, c.index.coupling_orders)
            for c in python.currents
            if not c.is_source
        }
    )
    requested = RequestedLoopBlocks(frozenset(rows[::3]))
    _assert_equal(
        compiled,
        model,
        "g g > g g g",
        {"QCD": 5},
        loop_block_reachability=requested,
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("text", "orders"),
    (
        ("g g > g g g g", {"QCD": 6}),
        ("u u~ > d d~ g g", {"QCD": 6}),
        ("u u~ > g g g g", {"QCD": 6}),
    ),
)
def test_native_loop_blocks_six_legs(models, text, orders):
    compiled, model = models
    _assert_equal(compiled, model, text, orders)


def test_stage_digest_names_its_stage():
    assert dag_input.stage_digest("S1'", "00" * 32, "11" * 32) != (
        dag_input.stage_digest("S1''", "00" * 32, "11" * 32)
    )
    with pytest.raises(ValueError, match="unknown stage"):
        dag_input.stage_digest("S2", "00" * 32, "11" * 32)
