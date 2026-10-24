# SPDX-License-Identifier: 0BSD
"""Native loop-request DAG builder against the tree engine (WP3.1 A1-A3).

The oracle is ``GenericDAGCompiler(loop_blocks=True)`` with
``loops.v2.blocks._blocks``: the native DAG must decode to an equal
``GenericDAG`` (same currents, sources and interactions in the same order),
have the same flat-record digest (every float bit), and group into equal
``Block`` records for each tree-kernel variant. ``RUST-MIGRATION.md`` §3.A.

A3 adds the CT program (``ct_closure.CountertermBlocks.build`` over the
CT-lowered model, DESIGN §6.1 items 1-6 and 8: its DAG, blocks and clone
lines, the U(1) leakage check) on the §6.1 block sets, and the
shared-contract value-equivalence derivation of
``assign_recursive_current_evaluation_reuse`` value by value against
``_derive_current_value_equivalences`` (A0's electroweak fixture with merged
classes, and copies scaled by inexact and complex factors).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.generation.dag_compiler import GenericDAGCompiler
from pyamplicol.generation.dag_equivalence import (
    _derive_current_value_equivalences,
    _shared_contract_current_ids,
    assign_recursive_current_evaluation_reuse,
)
from pyamplicol.loops.v2 import ct_closure, dag_input
from pyamplicol.loops.v2.blocks import (
    RequestedLoopBlocks,
    _blocks,
    vetoed_particle_ids,
    ward_breaking_fields,
)

from . import test_dag_equivalence_contract_classes as contract_classes
from .test_loop_v2_ct_two_point import two_point_model  # noqa: F401 (fixture)

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
    # No coloured leg: the colour-sector branch of the engine (A3).
    ("e+ e- > e+ e- a", {"QCD": 2, "QED": 3}),
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
def test_native_builder_needs_no_symbolica_licence(
    models, loop_sm_compiled_model, tmp_path
):
    """The Rust builder runs on its tables alone, in a process without a key:
    a loop-block DAG, and the CT program of g g > g g g, whose shared
    contracts run the derivation with its exact rationals.

    The child imports the extension, which starts a restricted Symbolica
    instance: like ``test_six_gluon_generation_completes_without_a_personal_key``
    it must not overlap another unlicensed Symbolica process of the user.
    """

    from pyamplicol.models.external import CompiledUFOModel

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    paths, expected = [], []
    paths.append(tmp_path / "blocks.json")
    paths[-1].write_text(
        dag_input.dag_input(process, model, max_coupling_orders={"QCD": 5})
    )
    expected.append(_native(model, process, {"QCD": 5}).digest())
    program, _native_program = _assert_ct_equal(
        loop_sm_compiled_model, "g g > g g g", {"QCD": 3}, {"QCD": 5}
    )
    dag = program.blocks.blocks_dag
    paths.append(tmp_path / "ct.json")
    paths[-1].write_text(
        dag_input.dag_input(
            dag.process,
            CompiledUFOModel(program.recursion.model.lowered.compiled),
            max_coupling_orders=program.recursion.coupling_order_limits(),
            reachability=ct_closure.RootReachability((1 << len(dag.process.legs)) - 1),
        )
    )
    expected.append(
        dag_input.NativeDag.from_dag(
            dag, CompiledUFOModel(program.recursion.model.lowered.compiled)
        ).digest()
    )
    environment = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    script = (
        "import sys\n"
        "from pyamplicol.loops.native import load_loop_native\n"
        "for path in sys.argv[1:]:\n"
        "    print(load_loop_native().V2Dag.build(open(path).read()).digest())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *map(str, paths)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert result.stdout.strip().splitlines()[-2:] == expected


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


# ------------------------------------------------------------------ A3


def _records(equivalences):
    return [
        (item.representative_id, item.factor[0].hex(), item.factor[1].hex())
        for item in equivalences
    ]


def _evaluations(interactions):
    return [
        (
            item.evaluation_group_id,
            item.evaluation_factor[0].hex(),
            item.evaluation_factor[1].hex(),
        )
        for item in interactions
    ]


def _assert_reuse_equal(dag, model):
    """The native derivation (full and restricted) and rewrite of ``dag``
    equal the oracle's bit for bit; returns the full equivalences."""

    full = _derive_current_value_equivalences(dag, model)
    native_full, _interactions = dag_input.native_reuse(dag, model, restricted=False)
    assert _records(native_full) == _records(full)
    restricted = _derive_current_value_equivalences(
        dag, model, derived_current_ids=_shared_contract_current_ids(dag)
    )
    native, interactions = dag_input.native_reuse(dag, model)
    assert _records(native) == _records(restricted)
    expected = assign_recursive_current_evaluation_reuse(dag, model).interactions
    assert interactions == expected
    assert _evaluations(interactions) == _evaluations(expected)
    return full


# Copies of the electroweak DAG scaled by exact non-unit, complex and inexact
# factors: projective classes through 3 + i/2 and 1/2 - i/4, sign classes,
# and products that are not binary64 values (0.1), which fall back to the
# concrete inputs.
SCALED_FACTORS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (3.0, 0.5),
    (0.1, 0.0),
    (0.5, -0.25),
    (0.0, -1.0),
)


@pytest.mark.loop_fast
def test_native_reuse_derivation_equals_the_oracle_bit_for_bit(monkeypatch):
    """A0's electroweak fixture with merged classes (factors ±1, ±i, 2, 4, 8
    and one deformed interaction) value by value, then copies scaled by the
    factors above (RUST-MIGRATION §3.A, A3 risk: bit-exact binary64)."""

    import json

    dag, model = contract_classes._sector_copies()
    full = _assert_reuse_equal(dag, model)
    fixture = json.loads(contract_classes.FIXTURE.read_text(encoding="utf-8"))
    assert [list(record) for record in _records(full)] == fixture["equivalences"]

    monkeypatch.setattr(contract_classes, "FACTORS", SCALED_FACTORS)
    dag, model = contract_classes._sector_copies()
    full = _assert_reuse_equal(dag, model)
    merged = {
        item.factor
        for index, item in enumerate(full)
        if item.representative_id != index
    }
    # Projective classes beyond the signs, with complex exact factors.
    assert any(re not in (0.0, 1.0, -1.0) and im != 0.0 for re, im in merged)
    assert (-1.0, 0.0) in merged and (0.0, -1.0) in merged


_PROGRAMS: dict = {}


def _python_program(compiled_path, text, born, loop, veto=()):
    """The production Python CT program of a request, once per module."""

    from .test_loop_v2_ct_recursion import _program_and_rules

    key = (str(compiled_path), text, repr(born), repr(loop), tuple(veto))
    if key not in _PROGRAMS:
        _PROGRAMS[key] = _program_and_rules(compiled_path, text, born, loop, veto)[0]
    return _PROGRAMS[key]


def _assert_ct_equal(compiled_path, text, born, loop, veto=(), *, rebuild=False):
    """The native CT program of a request equals the production Python one
    (``counterterm_program_blocks``): whole ``GenericDAG``, flat-record digest,
    blocks, clone lines, kernel variant, U(1) leakage, Ward query; with
    ``rebuild`` a second native build is byte-identical (rule 5: the builder
    is sequential, so no thread count enters)."""

    from pyamplicol.models.external import CompiledUFOModel

    program = _python_program(compiled_path, text, born, loop, veto)
    dag = program.blocks.blocks_dag
    lowered = CompiledUFOModel(program.recursion.model.lowered.compiled)
    ignored = vetoed_particle_ids(veto)
    native = dag_input.native_counterterm_blocks(
        program.recursion, dag.process, ignored_particle_ids=ignored
    )
    assert native.dag.decode() == dag
    assert native.dag.digest() == dag_input.NativeDag.from_dag(dag, lowered).digest()
    assert native.kernel_variant == program.blocks.kernel_variant
    assert native.blocks == program.blocks.blocks
    assert dict(native.clones) == dict(program.clones)
    assert dag_input.u1_leaking_structures(native) == (
        ct_closure.u1_leaking_structures(program)
    )
    assert native.dag.coloured_propagated() == _propagated(dag, lowered)
    if rebuild:
        again = dag_input.native_counterterm_blocks(
            program.recursion, dag.process, ignored_particle_ids=ignored
        )
        assert again.dag.columns() == native.dag.columns()
        assert again.blocks == native.blocks
    return program, native


CT_CONTROLS = (
    ("g g > g g g", {"QCD": 3}, {"QCD": 5}),
    ("g g > u u~", {"QCD": 2}, {"QCD": 4}),
)


@pytest.mark.loop_fast
@pytest.mark.parametrize(("text", "born", "loop"), CT_CONTROLS)
def test_native_counterterm_program_equals_the_closure(
    loop_sm_compiled_model, text, born, loop
):
    """The DESIGN §6.1 block set of the CT program: multi-trace colour
    states, U(1)-projection copies and clone lines of the lowered model."""

    program, native = _assert_ct_equal(
        loop_sm_compiled_model, text, born, loop, rebuild=text == "g g > g g g"
    )
    assert native.clones
    if text == "g g > g g g":
        # Closed colour components share evaluation contracts: the derivation.
        assert _shared_contract_current_ids(program.blocks.blocks_dag)
        assert any(
            c.index.color_state.traces for c in program.blocks.blocks_dag.currents
        )


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "born", "loop", "veto"),
    (
        ("u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}, ()),
        ("u u~ > d d~ g g", {"QCD": 4}, {"QCD": 6}, ()),
        ("g g > u u~ a g", {"QCD": 3, "QED": 1}, {"QCD": 5, "QED": 1}, ()),
        ("u u~ > a g g g", {"QCD": 3, "QED": 1}, {"QCD": 5, "QED": 1}, ()),
        ("g g > t t~ g g", {"QCD": 4}, {"QCD": 6}, ()),
        ("g g > g g g", {"QCD": 3}, {"QCD": 5}, (6,)),
        # No coloured leg (photon, Z and quark-loop two-point counterterms).
        ("e+ e- > e+ e-", {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2}, ()),
        ("u d~ > e+ ve g", {"QCD": 1, "QED": 2}, {"QCD": 3, "QED": 2}, ()),
    ),
)
def test_native_counterterm_program_block_sets(
    loop_sm_compiled_model, text, born, loop, veto
):
    """The remaining §6.1 block sets (the U(1) projection taking a quark
    line's gluon, photons, massive tops with W_T != 0) and a top veto."""

    _assert_ct_equal(loop_sm_compiled_model, text, born, loop, veto)


@pytest.mark.loop_physics
def test_native_u1_leakage_scan_equals_the_closure(loop_sm_compiled_model, monkeypatch):
    """The DAG scan of ``u1_leaking_structures`` on a program whose d-symbol
    vertices absorb and feed quark lines' gluons, with the projection gate
    off (loop_sm's lowered model is projected, so the check passes there and
    the production chain would refuse the request without it)."""

    program, native = _assert_ct_equal(
        loop_sm_compiled_model, "u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}
    )
    monkeypatch.setattr(ct_closure, "u1_projected", lambda compiled: False)
    leaking = ct_closure.u1_leaking_structures(program)
    assert leaking
    assert dag_input.u1_leaking_structures(native) == leaking


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "born", "loop"),
    (
        ("u u~ > e- e+", {"QCD": 0, "QED": 2}, {"QCD": 2, "QED": 2}),
        ("u u~ > e- e+ g", {"QCD": 1, "QED": 2}, {"QCD": 3, "QED": 2}),
        ("g g > u u~ g", {"QCD": 3, "QED": 0}, {"QCD": 5, "QED": 0}),
    ),
)
def test_native_counterterm_program_two_point_model(
    two_point_model,  # noqa: F811
    text,
    born,
    loop,
):
    """The two-point synthetic model (photon-Z mixing line, chiral and
    flipping u terms on a Dirac-forced u, d-s mixing; with a gluon, the
    g g Z R2 whose Levi-Civita cell cancels between the quark doublets and
    is left out of the lowered model, ``test_loop_v2_ct_cancellation``)."""

    _program, native = _assert_ct_equal(two_point_model, text, born, loop)
    assert native.clones


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("text", "born", "loop"),
    (
        ("g g > g g g g", {"QCD": 4}, {"QCD": 6}),
        ("u u~ > g g g g", {"QCD": 4}, {"QCD": 6}),
    ),
)
def test_native_counterterm_program_six_legs(loop_sm_compiled_model, text, born, loop):
    """The S1'' stage digest (CT DAG, blocks, clone lines) at six legs."""

    program, native = _assert_ct_equal(loop_sm_compiled_model, text, born, loop)
    from pyamplicol.models.external import CompiledUFOModel

    lowered = CompiledUFOModel(program.recursion.model.lowered.compiled)
    python = dag_input.stage_digest(
        "S1''",
        dag_input.NativeDag.from_dag(program.blocks.blocks_dag, lowered).digest(),
        dag_input.blocks_digest(program.blocks.blocks, program.clones),
    )
    assert python == dag_input.stage_digest(
        "S1''",
        native.dag.digest(),
        dag_input.blocks_digest(native.blocks, native.clones),
    )
