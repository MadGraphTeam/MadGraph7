# SPDX-License-Identifier: 0BSD
"""Native loop-request DAG builder against the tree engine (WP3.1 A1-A5).

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

A4 switches the production call sites (``dag_input.NATIVE_DAG``):
``build_tree_blocks`` and ``CountertermBlocks.build`` equal their oracle
stages without a decode (rule 9: none in generation, one in the export),
each listed fallback is refused for real and recorded (rule 2), 1 and N
threads give the same stages (rule 5). A5 checks the counterterm program's
native interaction program (``dag_input.NATIVE_PROGRAM``) against
``_CountertermTrees.of`` on the Python rows, byte for byte.
"""

from __future__ import annotations

import hashlib
import itertools
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
    propagated_species,
    vetoed_particle_ids,
    ward_breaking_fields,
)

from . import test_dag_equivalence_contract_classes as contract_classes
from .test_loop_v2_ct_two_point import two_point_model  # noqa: F401 (fixture)

VARIANTS = ("transverse", "full")


def _flag(on):
    """``dag_input.NATIVE_DAG`` set to ``on`` (off: the oracle stages)."""

    return dag_input.native_flags(dag=on)


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
    assert native.propagated_species() == propagated_species(python)
    return python, native


CONTROLS = (
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
def test_native_grade_split_equals_the_tree_engine(models):
    """Leading and next-to-leading colour grade the blocks whose trees mix
    N_c grades (``generation.dag_grades``): in u u~ > u u~ g the quark
    current of word (3, 1, 2) is reached through the gluon (1, 2) and through
    the U(1)-subtraction piece of (3, 1). Both lanes are equal, and every
    block has one grade (full colour keeps blocks that mix them). A request
    whose blocks have one grade each keeps its DAG (u u~ > d d~ g)."""

    compiled, model = models
    text, orders = "u u~ > u u~ g", {"QCD": 5}
    python, _native_dag = _assert_equal(compiled, model, text, orders, u1_grades=True)
    assert any(c.index.color_state.nc_grade is not None for c in python.currents)
    assert all(
        block.colour_flow.nc_grade is not None
        for block in _blocks(python, model, VARIANTS[0])
    )
    plain = _python(model, _process(compiled, text), orders)
    assert any(
        block.colour_flow.nc_grade is None
        for block in _blocks(plain, model, VARIANTS[0])
    )
    text = "u u~ > d d~ g"
    graded, _native_dag = _assert_equal(compiled, model, text, orders, u1_grades=True)
    assert graded == _python(model, _process(compiled, text), orders)


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
    from pyamplicol.generation.dag_compiler import compile_generic_dag
    from pyamplicol.models import BuiltinSMModel
    from pyamplicol.models.builtin.process_ir import build_process_ir

    # A full-colour tree DAG whose helicity ancestry needs 80 bits (rule 2).
    tree = compile_generic_dag(
        build_process_ir("u u~ > d d~ g", color_accuracy="full"),
        model=BuiltinSMModel(),
    )
    with pytest.raises(dag_input.NativeDagUnsupported, match="c_ancestry"):
        dag_input.encode_dag(tree)


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


def test_native_builder_is_sequential():
    """Rule 5: the builder, its derivation and its binding run on the calling
    thread (no Rayon pool, no spawned thread), so the columns are the same
    with 1 and N threads; ``test_native_build_is_deterministic`` repeats a
    build. (The seven-leg S1'/S1'' records of the A4 report ran with eight
    threads and equal the Python oracle's.)"""

    root = Path(__file__).resolve().parents[2] / "rust"
    sources = [
        *sorted((root / "crates/rusticol-core/src/loop_v2/dag").glob("*.rs")),
        root / "loop-reduction/python/src/v2_dag.rs",
    ]
    for path in sources:
        text = path.read_text()
        for name in ("rayon", "par_iter", "into_par_iter", "thread::spawn", "scope("):
            assert name not in text, (path, name)


# The A5 bindings' outputs as JSON values (the no-key child runs the same).
_PROGRAM_PROBE = (
    "def program_record(handle, text):\n"
    "    program = handle.program(text)\n"
    "    starts, variants = program.slots()\n"
    "    texts = [('re%d' % k, 'im%d' % k) for k in range(len(program.weights()))]\n"
    "    uses = program.kernels()\n"
    "    vertex = [item for kind, _key, item in uses if kind == 'kernels']\n"
    "    record = {\n"
    "        'species': handle.current_species(),\n"
    "        'value_count': program.value_count(),\n"
    "        'masks': program.momentum_masks(),\n"
    "        'slots': [bytes(starts).hex(), bytes(variants).hex()],\n"
    "        'kernels': uses,\n"
    "        'weights': program.weights(),\n"
    "        'signatures': program.signatures(),\n"
    "        'particles': program.particles(),\n"
    "        'steps': program.steps(texts),\n"
    "        'sources': handle.source_rows(),\n"
    "        'interactions': handle.interaction_rows(vertex + program.signatures()),\n"
    "    }\n"
    "    return json.loads(json.dumps(record, default=list))\n"
)


def _portability_tool():
    """``tools/loop_artifact_portability.py`` (its ``RestrictedPermit``)."""

    import importlib.util

    path = (
        Path(__file__).resolve().parents[2] / "tools" / "loop_artifact_portability.py"
    )
    spec = importlib.util.spec_from_file_location("_native_dag_portability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.loop_nightly
def test_native_builder_needs_no_symbolica_licence(
    models, loop_sm_compiled_model, tmp_path, monkeypatch
):
    """Rule 4: every binding runs on its tables alone in a process without a
    key, and equals the oracle there. ``V2Dag.build``: a loop-block DAG and
    the CT program of g g > g g g (its closed colour components share
    contracts but merge no class, so the build's derivation sees unit
    factors only). ``V2Dag.reuse``, full and restricted, with the rewrite:
    the electroweak copies scaled by ``SCALED_FACTORS`` and the guard copies,
    whose non-unit factor products and quotients take the Symbolica
    ``Rational`` path of ``dag/exact.rs``. ``V2Dag.u1_leaking``: the leaking
    CT DAG of u u~ > d d~ g (the projection gate off, as in the
    ``loop_physics`` test).

    The child runs while another process holds the restricted Symbolica
    permit of a private lock directory (``RestrictedPermit`` of
    ``tools/loop_artifact_portability.py``), with pyAmpliCol's native library
    key off: a binding that touched Symbolica state would abort, and no
    other unlicensed Symbolica process of the user (whose permit lives in
    the default lock directory) can clash with it.
    """

    import json

    from pyamplicol.generation import dag_equivalence
    from pyamplicol.models.external import CompiledUFOModel

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    request: dict = {"build": [], "reuse": []}
    expected: dict = {"digests": [], "reuse": [], "interactions": []}
    request["build"].append(str(tmp_path / "blocks.json"))
    Path(request["build"][-1]).write_text(
        dag_input.dag_input(process, model, max_coupling_orders={"QCD": 5})
    )
    loop_blocks = _native(model, process, {"QCD": 5})
    expected["digests"].append(loop_blocks.digest())
    # A4's bindings: the blocks with their interned helicity assignments and
    # the propagated species (Ward and coloured-width queries).
    request["blocks"] = str(tmp_path / "block-tables.json")
    Path(request["blocks"]).write_text(loop_blocks._tables())
    # ``Vec<u8>`` fields (words, traces) arrive as ``bytes``, which the
    # decoder reads as int sequences: compared as lists on both sides.
    expected["blocks"] = json.loads(
        json.dumps(loop_blocks._handle.blocks(loop_blocks._tables()), default=list)
    )
    full = (1 << len(process.legs)) - 1
    request["full"] = full
    expected["species"] = [
        list(item) for item in loop_blocks._handle.propagated_species(full)
    ]
    program, _native_program = _assert_ct_equal(
        loop_sm_compiled_model, "g g > g g g", {"QCD": 3}, {"QCD": 5}
    )
    dag = program.blocks.blocks_dag
    lowered = CompiledUFOModel(program.recursion.model.lowered.compiled)
    request["build"].append(str(tmp_path / "ct.json"))
    Path(request["build"][-1]).write_text(
        dag_input.dag_input(
            dag.process,
            lowered,
            max_coupling_orders=program.recursion.coupling_order_limits(),
            reachability=ct_closure.RootReachability((1 << len(dag.process.legs)) - 1),
        )
    )
    expected["digests"].append(dag_input.NativeDag.from_dag(dag, lowered).digest())

    # The oracle's exact helpers on non-unit factors: the twins in exact.rs
    # take the Rational path on the same arguments.
    exact_calls = {"product": 0, "ratio": 0}
    product = dag_equivalence._exact_representable_complex_product
    ratio = dag_equivalence._exact_representable_complex_ratio

    def counted_product(left, right):
        if left not in dag_equivalence._EXACT_UNITS and (
            right not in dag_equivalence._EXACT_UNITS
        ):
            exact_calls["product"] += 1
        return product(left, right)

    def counted_ratio(numerator, denominator):
        exact_calls["ratio"] += 1
        return ratio(numerator, denominator)

    monkeypatch.setattr(contract_classes, "FACTORS", SCALED_FACTORS)
    fixtures = [contract_classes._sector_copies(), _guard_copies(monkeypatch)[:2]]
    for number, (reuse_dag, reuse_model) in enumerate(fixtures):
        with monkeypatch.context() as patch:
            patch.setattr(
                dag_equivalence, "_exact_representable_complex_product", counted_product
            )
            patch.setattr(
                dag_equivalence, "_exact_representable_complex_ratio", counted_ratio
            )
            records = [
                [
                    [
                        item.representative_id,
                        dag_input._bits(item.factor[0]),
                        dag_input._bits(item.factor[1]),
                    ]
                    for item in _derive_current_value_equivalences(
                        reuse_dag, reuse_model, derived_current_ids=derived
                    )
                ]
                for derived in (None, _shared_contract_current_ids(reuse_dag))
            ]
            rewritten = assign_recursive_current_evaluation_reuse(
                reuse_dag, reuse_model
            )
        paths = [tmp_path / f"reuse{number}.{suffix}" for suffix in ("bin", "json")]
        paths[0].write_bytes(dag_input.encode_dag(reuse_dag))
        paths[1].write_text(dag_input.reuse_tables(reuse_dag, reuse_model))
        request["reuse"].append([*map(str, paths), str(tmp_path / f"out{number}.bin")])
        expected["reuse"].append(records)
        expected["interactions"].append(rewritten.interactions)
    assert exact_calls["product"] > 0 and exact_calls["ratio"] > 0

    program, native = _assert_ct_equal(
        loop_sm_compiled_model, "u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}
    )
    monkeypatch.setattr(ct_closure, "u1_projected", lambda compiled: False)
    scans = []
    scan = dag_input.NativeDag.u1_leaking

    def recorded(self, kinds, generator_kinds, adjoint, fundamental, rest):
        arguments = [
            *(
                sorted(int(v) for v in values)
                for values in (kinds, generator_kinds, adjoint)
            ),
            int(fundamental),
            int(rest),
        ]
        found = scan(self, *arguments)
        scans.append((arguments, sorted(found)))
        return found

    monkeypatch.setattr(dag_input.NativeDag, "u1_leaking", recorded)
    assert ct_closure._u1_leaking(native.recursion, native.dag) == (
        ct_closure.u1_leaking_structures(program)
    )
    ((arguments, leaking),) = scans
    assert leaking
    (tmp_path / "u1.bin").write_bytes(native.dag.columns())
    request["u1"] = [str(tmp_path / "u1.bin"), arguments]
    # A5's bindings: the interaction program of the CT DAG (its slot masks)
    # and the source and interaction rows the export reads.
    from pyamplicol.loops.v2.native_trees import counterterm_slots

    lowered = CompiledUFOModel(program.recursion.model.lowered.compiled)
    slot_orders = [slot[0] for slot in counterterm_slots(program.recursion)]
    assert any(name.endswith("_EPS1") for name in slot_orders)
    text, _payloads = dag_input._program_input(native.dag, lowered, slot_orders)
    request["program"] = [str(tmp_path / "u1.bin"), text]
    probe: dict = {"json": json}
    exec(_PROGRAM_PROBE, probe)
    expected["program"] = probe["program_record"](native.dag._native_handle(), text)
    # Every step records its currents' slot sets; some reach one slot only.
    assert all("accumulator_masks" in step for step in expected["program"]["steps"])
    assert any(
        any(mask > 0 for mask in step["accumulator_masks"])
        for step in expected["program"]["steps"]
    )
    (tmp_path / "request.json").write_text(json.dumps(request))

    script = (
        "import json, sys\n"
        "from pyamplicol.loops.native import load_loop_native\n"
        "native = load_loop_native()\n"
        "request = json.load(open(sys.argv[1]))\n"
        "out = {'digests': [], 'reuse': []}\n"
        "for path in request['build']:\n"
        "    out['digests'].append(native.V2Dag.build(open(path).read()).digest())\n"
        "handle = native.V2Dag.build(open(request['build'][0]).read())\n"
        "tables = open(request['blocks']).read()\n"
        "out['blocks'] = json.loads(json.dumps(handle.blocks(tables), default=list))\n"
        "species = handle.propagated_species(request['full'])\n"
        "out['species'] = [list(x) for x in species]\n"
        "for columns, tables, rewritten in request['reuse']:\n"
        "    handle = native.V2Dag.from_columns(open(columns, 'rb').read())\n"
        "    records = []\n"
        "    for restricted in (False, True):\n"
        "        found, dag = handle.reuse(open(tables).read(), restricted)\n"
        "        records.append([list(item) for item in found])\n"
        "    open(rewritten, 'wb').write(bytes(dag.columns()))\n"
        "    out['reuse'].append(records)\n"
        "columns, arguments = request['u1']\n"
        "handle = native.V2Dag.from_columns(open(columns, 'rb').read())\n"
        "out['u1'] = sorted(handle.u1_leaking(*arguments))\n"
        + _PROGRAM_PROBE
        + "columns, text = request['program']\n"
        "handle = native.V2Dag.from_columns(open(columns, 'rb').read())\n"
        "out['program'] = program_record(handle, text)\n"
        "print(json.dumps(out))\n"
    )
    environment = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    # pyAmpliCol's library key off in the native code (crate ``pyamplicol``):
    # its copy of Symbolica is in restricted mode.
    environment["PYAMPLICOL_NATIVE_LIBRARY_KEY"] = "off"
    portability = _portability_tool()
    lock = tmp_path / "symbolica-locks"
    lock.mkdir(mode=0o700)
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "request.json")],
            env=permit.environment(environment),
            capture_output=True,
            text=True,
            timeout=300,
        )
    # The positive control: a plain Symbolica process aborted meanwhile.
    assert permit.record["verified"], permit.record
    if result.returncode != 0:
        restricted = any(
            marker in result.stdout + result.stderr
            for marker in portability.RESTRICTED_MARKERS
        )
        pytest.fail(
            f"the unlicensed child exited with {result.returncode}"
            + (
                " in Symbolica's restricted mode: a binding touched Symbolica "
                "state (rule 4)"
                if restricted
                else ""
            )
            + f"; its permit is private ({lock}), so no other process clashed"
            + f"\nstdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["digests"] == expected["digests"]
    assert out["blocks"] == expected["blocks"]
    assert out["species"] == expected["species"]
    assert out["reuse"] == expected["reuse"]
    for (reuse_dag, _model), (*_inputs, rewritten), interactions in zip(
        fixtures, request["reuse"], expected["interactions"], strict=True
    ):
        decoded = dag_input._decode(
            Path(rewritten).read_bytes(), reuse_dag.process, reuse_dag.color_plan
        )
        assert decoded.interactions == interactions
    assert out["u1"] == leaking
    assert out["program"] == expected["program"]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "orders"),
    (
        ("g g > g g", {"QCD": 4}),
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
    assert (int(top.pdg_code), False) in native.propagated_species()
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
    ("text", "orders", "u1_grades"),
    (
        ("g g > g g g g", {"QCD": 6}, False),
        ("u u~ > d d~ g g", {"QCD": 6}, False),
        ("u u~ > g g g g", {"QCD": 6}, False),
        # Three identical-flavour lines, graded (grades 0, -1 and -2).
        ("u u~ > u u~ u u~", {"QCD": 6}, True),
    ),
)
def test_native_loop_blocks_six_legs(models, text, orders, u1_grades):
    compiled, model = models
    python, _native_dag = _assert_equal(
        compiled, model, text, orders, u1_grades=u1_grades
    )
    if u1_grades:
        grades = {c.index.color_state.nc_grade for c in python.currents}
        assert {0, -1, -2} <= grades, grades


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


# Coefficients of the guard copies (``_guard_copies``): ``c / 3`` does not
# round-trip (``3 * fl(c / 3) != c``); ``p / 3`` and ``q / 9`` round to the same
# lookup key and round-trip, but ``3 * p != q``.
GUARD_NON_ROUNDTRIP = float.fromhex("0x1.f4bea97b9e977p+0")
GUARD_UNSCALED = (
    float.fromhex("0x1.414c3423c5fd7p+0"),
    float.fromhex("0x1.e1f24e35a8fc3p+1"),
)


def _rounded_ratio(numerator, denominator):
    quotient = complex(*numerator) / complex(*denominator)
    return (quotient.real, quotient.imag)


# Each mutant drops one fail-closed guard of the oracle (``dag_equivalence``):
# the exactness proof of the projective factor, ``fsum``, the round-trip bit
# check of the lookup key, the coefficient-by-coefficient scaling check.
GUARD_MUTANTS = {
    "_exact_representable_complex_ratio": _rounded_ratio,
    "fsum": sum,
    "_roundtrip_complex_ratio": _rounded_ratio,
    "_term_vector_scaled_exactly": lambda representative, factor, candidate: True,
}


def _guard_copies(monkeypatch):
    """A0's electroweak DAG with four exact copies whose coefficients decide
    one class at each guard of ``GUARD_MUTANTS``; returns the DAG, the model
    and the current each guard decides.

    Copy 1: a one-term current is 3 in every other copy and 1 here; the
    lookup keys collide, and 1/3 is not a binary64 value (its rounding would
    scale 3 back to 1 exactly). Copy 2: one term summed over three
    interactions 2**60, 1, -2**60 (``fsum`` 1: the class merges; a plain sum
    0). Copy 3: a two-term current (3, c) everywhere and (6, 2c) here, with
    no lookup key (c / 3 does not round-trip). Copy 4: (3, p) and (9, q),
    the same key and an exact pivot ratio 3, but 3p != q."""

    from collections import defaultdict
    from dataclasses import replace

    from pyamplicol.generation.dag_equivalence import _current_term_vector

    copies = 4
    monkeypatch.setattr(contract_classes, "FACTORS", ((1.0, 0.0),) * copies)
    monkeypatch.setattr(contract_classes, "DEFORMATION", (1.0, 0.0))
    dag, model = contract_classes._sector_copies()
    count = len(dag.currents) // (copies + 1)
    stride = len(dag.interactions) // (copies + 1)
    by_result = defaultdict(list)
    for item in dag.interactions[:stride]:
        by_result[item.result_id].append(item.id)
    generated = [c.id for c in dag.currents[:count] if not c.is_source]
    singles = [
        i
        for i in generated
        if len(dag.currents[i].index.external_labels) == 2 and len(by_result[i]) == 1
    ]
    pairs = [
        i
        for i in generated
        if len(dag.currents[i].index.external_labels) == 3 and len(by_result[i]) == 2
    ]
    equivalences = list(_derive_current_value_equivalences(dag, model))

    def in_term_order(current_id):
        # The interactions of a two-term current in its term-vector order
        # (unit evaluation factors: the coefficients are the colour weights).
        probe = [
            replace(dag.interactions[i], color_weight=(float(k + 1), 0.0))
            for k, i in enumerate(by_result[current_id])
        ]
        terms = _current_term_vector(
            dag,
            dag.currents[current_id],
            probe,
            model,
            current_equivalences=equivalences,
            equivalence_by_kind={},
        )
        values = [coefficient for _key, coefficient in terms]
        assert sorted(values) == [(1.0, 0.0), (2.0, 0.0)]
        return [by_result[current_id][int(value[0]) - 1] for value in values]

    weights = {}
    rows = (
        (singles[0], (3.0,), (1.0,), 1),
        (pairs[0], (3.0, GUARD_NON_ROUNDTRIP), (6.0, 2 * GUARD_NON_ROUNDTRIP), 3),
        (pairs[1], (3.0, GUARD_UNSCALED[0]), (9.0, GUARD_UNSCALED[1]), 4),
    )
    for current_id, everywhere, here, special in rows:
        ids = (
            in_term_order(current_id) if len(everywhere) > 1 else by_result[current_id]
        )
        for copy in range(copies + 1):
            for item, value in zip(
                ids, here if copy == special else everywhere, strict=True
            ):
                weights[item + copy * stride] = (value, 0.0)
    interactions = [
        replace(item, color_weight=weights.get(item.id, item.color_weight))
        for item in dag.interactions
    ]
    (summed,) = by_result[singles[1]]
    summed += 2 * stride
    interactions[summed] = replace(interactions[summed], color_weight=(2.0**60, 0.0))
    for value in (1.0, -(2.0**60)):
        interactions.append(
            replace(
                interactions[summed], id=len(interactions), color_weight=(value, 0.0)
            )
        )
    decided = {
        "_exact_representable_complex_ratio": singles[0] + count,
        "fsum": singles[1] + 2 * count,
        "_roundtrip_complex_ratio": pairs[0] + 3 * count,
        "_term_vector_scaled_exactly": pairs[1] + 4 * count,
    }
    return replace(dag, interactions=tuple(interactions)), model, decided


@pytest.mark.loop_fast
def test_native_reuse_derivation_fails_closed_where_the_oracle_does(monkeypatch):
    """The derivation's fail-closed guards (``_guard_copies``) value by value:
    each decides one class, which moves when the oracle drops that guard, so
    the native twins (``dag/exact.rs`` ``exact_ratio``, ``roundtrip_ratio``,
    ``fsum``; ``reuse.rs`` ``scaled_exactly``) are pinned there."""

    from pyamplicol.generation import dag_equivalence

    dag, model, decided = _guard_copies(monkeypatch)
    full = _assert_reuse_equal(dag, model)
    for name, current_id in decided.items():
        # fsum merges the summed current with its original; every other
        # guard keeps its current in a class of its own.
        assert (full[current_id].representative_id == current_id) == (name != "fsum")
        with monkeypatch.context() as patch:
            patch.setattr(dag_equivalence, name, GUARD_MUTANTS[name])
            mutated = _derive_current_value_equivalences(dag, model)
        assert mutated[current_id] != full[current_id], name


_PROGRAMS: dict = {}


def _python_program(compiled_path, text, born, loop, veto=(), *, u1_grades=False):
    """The production Python CT program of a request, once per module."""

    from .test_loop_v2_ct_recursion import _program_and_rules

    key = (str(compiled_path), text, repr(born), repr(loop), tuple(veto), u1_grades)
    if key not in _PROGRAMS:
        with _flag(False):
            _PROGRAMS[key] = _program_and_rules(
                compiled_path, text, born, loop, veto, u1_grades=u1_grades
            )[0]
    return _PROGRAMS[key]


def _assert_ct_equal(
    compiled_path, text, born, loop, veto=(), *, rebuild=False, u1_grades=False
):
    """The native CT program of a request equals the production Python one
    (``counterterm_program_blocks``; ``u1_grades`` as for LC and NLC): whole
    ``GenericDAG``, flat-record digest, blocks, clone lines, kernel variant,
    U(1) leakage, Ward query; with ``rebuild`` a second native build is
    byte-identical (rule 5: the builder is sequential, so no thread count
    enters)."""

    from pyamplicol.models.external import CompiledUFOModel

    program = _python_program(
        compiled_path, text, born, loop, veto, u1_grades=u1_grades
    )
    dag = program.blocks.blocks_dag
    lowered = CompiledUFOModel(program.recursion.model.lowered.compiled)
    ignored = vetoed_particle_ids(veto)
    native = dag_input.native_counterterm_blocks(
        program.recursion,
        dag.process,
        ignored_particle_ids=ignored,
        u1_grades=u1_grades,
    )
    assert native.dag.decode() == dag
    assert native.dag.digest() == dag_input.NativeDag.from_dag(dag, lowered).digest()
    assert native.kernel_variant == program.blocks.kernel_variant
    assert native.blocks == program.blocks.blocks
    assert dict(native.clones) == dict(program.clones)
    assert ct_closure._u1_leaking(native.recursion, native.dag) == (
        ct_closure.u1_leaking_structures(program)
    )
    assert native.dag.propagated_species() == propagated_species(dag)
    if rebuild:
        again = dag_input.native_counterterm_blocks(
            program.recursion,
            dag.process,
            ignored_particle_ids=ignored,
            u1_grades=u1_grades,
        )
        assert again.dag.columns() == native.dag.columns()
        assert again.blocks == native.blocks
    return program, native


@pytest.mark.loop_fast
def test_native_counterterm_program_equals_the_closure(loop_sm_compiled_model):
    """The DESIGN §6.1 block set of the g g > g g g CT program: multi-trace
    colour states, U(1)-projection copies and clone lines of the lowered
    model (quark lines and the wider sets: ``loop_physics`` below)."""

    program, native = _assert_ct_equal(
        loop_sm_compiled_model, "g g > g g g", {"QCD": 3}, {"QCD": 5}, rebuild=True
    )
    assert native.clones
    # Closed colour components share evaluation contracts: the derivation.
    assert _shared_contract_current_ids(program.blocks.blocks_dag)
    assert any(c.index.color_state.traces for c in program.blocks.blocks_dag.currents)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "born", "loop", "veto"),
    (
        ("g g > u u~", {"QCD": 2}, {"QCD": 4}, ()),
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
    """The remaining §6.1 block sets (quark loops and lines, the U(1)
    projection taking a quark line's gluon, photons, massive tops with
    W_T != 0) and a top veto; and A5's interaction program of each (massive
    and chiral propagators, quark and photon kernels)."""

    program, _native = _assert_ct_equal(loop_sm_compiled_model, text, born, loop, veto)
    with _flag(True):
        blocks = ct_closure.CountertermBlocks.build(
            program.recursion,
            program.blocks.blocks_dag.process,
            ignored_particle_ids=vetoed_particle_ids(veto),
        )
    _assert_counterterm_trees_equal(program, blocks)


@pytest.mark.loop_physics
def test_native_counterterm_program_folds_mixed_connection_parities(
    loop_sm_compiled_model,
):
    """Three identical-flavour quark lines: a CT current closing the gluons
    of the pairing (a, b~)(b, a~) and one taking the U(1)-subtraction
    auxiliary of the pairing (a, a~)(b, b~) share their closed lines, so
    one block has derivations of both colour-connection parities. Both
    lanes fold the other parity's -1 into the same derivations' colour
    weights (``apply_colour_connection_signs``, ``signs::apply_connection_
    signs``): the native CT program equals the Python one, whole DAG and
    blocks (``_blocks`` refuses a block of mixed parities). The two
    derivations differ in N_c grade: with the grade split of leading and
    next-to-leading colour (``generation.dag_grades``, both lanes equal)
    they are separate blocks, and no block mixes parities."""

    from pyamplicol.generation.fermion_ordering import colour_connection_parities
    from pyamplicol.models.external import CompiledUFOModel

    request = ("u u~ > u u~ u u~", {"QCD": 4}, {"QCD": 6})
    program, _native = _assert_ct_equal(loop_sm_compiled_model, *request)
    dag = program.blocks.blocks_dag
    lowered = CompiledUFOModel(program.recursion.model.lowered.compiled)
    _parity, deviating = colour_connection_parities(dag, lowered)
    assert deviating
    assert all(current.index.color_state.nc_grade is None for current in dag.currents)
    graded, _native = _assert_ct_equal(loop_sm_compiled_model, *request, u1_grades=True)
    dag = graded.blocks.blocks_dag
    assert any(
        current.index.color_state.nc_grade is not None for current in dag.currents
    )
    _parity, deviating = colour_connection_parities(dag, lowered)
    assert not deviating


@pytest.mark.loop_physics
@pytest.mark.loop_physics
def test_native_counterterm_program_grades_are_order_free(loop_sm_compiled_model):
    """Full colour of g g > u u~ u u~ (no grade split): its CT program has
    currents derived first with grade -1 and then from a current of mixed
    grade. Both lanes give such a current no grade, whatever the order of
    its derivations (``blocks.u1_grades``, the single-grade view of
    ``dag_grades.u1_grade_sets``; ``fold_u1_grades`` natively), so every
    block's ``nc_grade`` is that of the order-free grade sets of its rows:
    one grade, or ``None``. Before, both lanes kept the first derivation's
    grade (32 currents of this program)."""

    from pyamplicol.generation.dag_grades import u1_grade_sets

    program, native = _assert_ct_equal(
        loop_sm_compiled_model, "g g > u u~ u u~", {"QCD": 4}, {"QCD": 6}
    )
    sets = u1_grade_sets(program.blocks.blocks_dag)
    mixed = 0
    for block in native.blocks:
        rows = [sets[row.current_id] for row in block.rows]
        grades = frozenset().union(*rows)
        single = len(grades) == 1 and all(len(found) == 1 for found in rows)
        expected = next(iter(grades)) if single else None
        assert block.colour_flow.nc_grade == expected, block.key.word
        mixed += expected is None
    assert mixed


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
    assert ct_closure._u1_leaking(native.recursion, native.dag) == leaking


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


# ------------------------------------------------------------------ A4


def _tree_blocks(model, process, orders, on, **options):
    from pyamplicol.loops.v2.blocks import build_tree_blocks

    with _flag(on):
        return build_tree_blocks(
            process,
            model,
            born_coupling_orders=options.pop("born", None),
            block_coupling_orders=orders,
            **options,
        )


def _assert_switch_equal(compiled, model, text, orders, **options):
    """``build_tree_blocks`` with the native stage equals the oracle stage:
    blocks, Born, S1' digest, the Ward-kernel and coloured-width queries,
    all without a decode; then reading a ``GenericDAG`` attribute of the
    handle decodes it once, to the oracle's DAG."""

    from pyamplicol.loops.v2 import snapshot
    from pyamplicol.loops.v2.artifact import coloured_widths

    process = _process(compiled, text)
    python = _tree_blocks(model, process, orders, False, **dict(options))
    before = dag_input.decode_count()
    native = _tree_blocks(model, process, orders, True, **dict(options))
    assert isinstance(native.blocks_dag, dag_input.NativeDag)
    assert not isinstance(python.blocks_dag, dag_input.NativeDag)
    assert native.dag_fallback is None
    assert native.blocks == python.blocks
    assert native.born_roots == python.born_roots
    assert native.kernel_variant == python.kernel_variant
    assert snapshot.loop_blocks_digest(native) == snapshot.loop_blocks_digest(python)
    assert propagated_species(native.blocks_dag) == propagated_species(
        python.blocks_dag
    )
    for fixed in ({}, {"WT": "0"}):
        native_dags = (native.blocks_dag, native.born_dag)
        python_dags = (python.blocks_dag, python.born_dag)
        assert ward_breaking_fields(native_dags, compiled, fixed) == (
            ward_breaking_fields(python_dags, compiled, fixed)
        )
        assert coloured_widths(native_dags, compiled, fixed) == (
            coloured_widths(python_dags, compiled, fixed)
        )
    full = native.with_kernels(model)
    assert full.blocks_dag is native.blocks_dag
    assert full.blocks == python.with_kernels(model).blocks
    assert dag_input.decode_count() == before
    # Rule 9: the handle decodes on the first attribute read, once.
    assert native.blocks_dag.currents == python.blocks_dag.currents
    assert dag_input.decode_count() == before + 1
    assert native.blocks_dag.materialize() == python.blocks_dag
    assert native.blocks_dag.interactions == python.blocks_dag.interactions
    assert dag_input.decode_count() == before + 1
    return python, native


def _native_progress(events, stage, dag, blocks):
    """The native stage's events among a ``progress_callback``'s
    (``dag_input.native_progress``): its start, then its finish with the
    counts."""

    currents, interactions, _sources = dag.counts()
    assert [event for event in events if event["step"] == "native-dag"] == [
        {"step": "native-dag", "stage": stage, "state": "start"},
        {
            "step": "native-dag",
            "stage": stage,
            "state": "done",
            "current_count": currents,
            "interaction_count": interactions,
            "block_count": len(blocks),
        },
    ]


@pytest.mark.loop_fast
def test_switched_tree_blocks_equal_the_oracle_stage(models):
    """A4: ``build_tree_blocks`` takes the native builder under
    ``NATIVE_DAG`` (u u~ > d d~ g: quark lines, connection signs), and
    reports the native stage's start and finish to ``progress_callback``."""

    compiled, model = models
    _python_route, native = _assert_switch_equal(
        compiled, model, "u u~ > d d~ g", {"QCD": 5}, born={"QCD": 3}
    )
    events: list = []
    again = _tree_blocks(
        model,
        _process(compiled, "u u~ > d d~ g"),
        {"QCD": 5},
        True,
        born={"QCD": 3},
        progress_callback=events.append,
    )
    assert again.blocks == native.blocks
    _native_progress(events, "loop-blocks", again.blocks_dag, again.blocks)


@pytest.mark.loop_fast
def test_switched_counterterm_program_equals_the_oracle_stage(
    loop_sm_compiled_model, monkeypatch
):
    """A4: ``CountertermBlocks.build`` takes the native builder under
    ``NATIVE_DAG``: the S1'' digest (DAG, blocks, clone lines) equals the
    oracle stage's, the U(1) leakage query runs natively (equal with the
    projection gate off too) and nothing is decoded. (The whole production
    chain under the flag: ``test_switched_generation_is_output_identical_``
    ``and_decodes_nothing``.)"""

    from pyamplicol.loops.v2 import snapshot

    # The program of the A3 default-tier test (built once per module).
    python = _python_program(
        loop_sm_compiled_model, "g g > g g g", {"QCD": 3}, {"QCD": 5}
    )
    before = dag_input.decode_count()
    events: list = []
    with _flag(True):
        native = ct_closure.CountertermBlocks.build(
            python.recursion,
            python.blocks.blocks_dag.process,
            progress_callback=events.append,
        )
    assert isinstance(native.blocks.blocks_dag, dag_input.NativeDag)
    assert native.blocks.dag_fallback is None
    _native_progress(
        events, "counterterm-program", native.blocks.blocks_dag, native.blocks.blocks
    )
    assert snapshot.counterterm_program_digest(native) == (
        snapshot.counterterm_program_digest(python)
    )
    assert native.blocks.blocks == python.blocks.blocks
    assert native.blocks.kernel_variant == python.blocks.kernel_variant
    assert dict(native.clones) == dict(python.clones)
    # The U(1) leakage query dispatches on the DAG (with the projection gate
    # off, quark lines leak: ``test_native_u1_leakage_scan_equals_the_closure``).
    for projected in (True, False):
        with monkeypatch.context() as patch:
            if not projected:
                patch.setattr(ct_closure, "u1_projected", lambda compiled: False)
            assert ct_closure.u1_leaking_structures(native) == (
                ct_closure.u1_leaking_structures(python)
            )
    assert dag_input.decode_count() == before


def _assert_counterterm_trees_equal(python_blocks, native_blocks):
    """A5: the native lane's counterterm program of a native CT DAG
    (``_CountertermTrees.of`` under ``NATIVE_PROGRAM``) equals the oracle's
    (``TreeRows`` and ``_interaction_program`` on the decoded DAG): the
    record (interaction program, layout, source fill), the saved kernels
    byte for byte, the value slots and what ``_parameter_program`` reads;
    nothing is decoded and released columns are released again."""

    import json
    from collections.abc import Mapping
    from dataclasses import fields

    from pyamplicol.loops.v2 import snapshot
    from pyamplicol.loops.v2.ct_closure import CountertermProgram
    from pyamplicol.loops.v2.native_trees import _CountertermTrees

    # The root sites do not enter the tree program.
    python = CountertermProgram(python_blocks, ())
    program = CountertermProgram(native_blocks, ())
    dag = program.blocks.blocks.blocks_dag
    assert type(dag) is dag_input.NativeDag
    dag.release()
    before = (dag_input.decode_count(), dag_input.rebuild_count())
    native = _CountertermTrees.of(program, 1234)
    assert dag_input.decode_count() == before[0]
    assert dag_input.rebuild_count() == before[1] + 1
    assert dag.released and not dag.decoded
    oracle = _CountertermTrees.of(python, 1234)
    assert json.dumps(native.record) == json.dumps(oracle.record)
    assert snapshot.counterterm_trees_digest(native) == (
        snapshot.counterterm_trees_digest(oracle)
    )
    # Every field ``of`` fills (a field added there is compared too): the
    # blocks are the programs' own objects, the rows compared below.
    for field in fields(_CountertermTrees):
        mine, theirs = getattr(native, field.name), getattr(oracle, field.name)
        if field.name == "blocks":
            assert len(mine) == len(theirs)
        elif field.name == "record":
            continue
        elif field.name != "rows":
            if isinstance(mine, Mapping):
                mine, theirs = dict(mine), dict(theirs)
            assert mine == theirs, field.name
    for name in ("_records", "_definitions", "_value_count", "_offset", "_size"):
        assert getattr(native.rows, name) == getattr(oracle.rows, name), name
    assert native.rows.compiled.name == oracle.rows.compiled.name
    return native, oracle


@pytest.mark.loop_fast
def test_native_counterterm_trees_equal_the_oracle(loop_sm_compiled_model):
    """A5 on the g g > g g g CT program (multi-trace colour states, clone
    lines, pole slots)."""

    python = _python_program(
        loop_sm_compiled_model, "g g > g g g", {"QCD": 3}, {"QCD": 5}
    )
    with _flag(True):
        blocks = ct_closure.CountertermBlocks.build(
            python.recursion, python.blocks.blocks_dag.process
        )
    native, _oracle = _assert_counterterm_trees_equal(python, blocks)
    # The slot table and the per-slot reachability masks (I5): CT = 0
    # currents reach every slot, a current with one insertion its slot.
    record = native.record
    assert [slot[0] for slot in record["slots"]] == [
        ct_closure.counterterm_slot_order(*slot) for slot in python.recursion.slots
    ]
    count = len(record["slots"])
    assert record["interactions"]["masks"] == [
        list(range(count)),
        *([s] for s in range(count)),
    ]
    steps = record["interactions"]["steps"]
    assert {mask for step in steps for mask in step["accumulator_masks"]} == set(
        range(count + 1)
    )
    # Off, or on a DAG the tree engine built, the oracle runs.
    with _program_flag(False):
        program = ct_closure.CountertermProgram(blocks, ())
        assert dag_input.native_counterterm_trees(program) is None
    program = ct_closure.CountertermProgram(python, ())
    assert dag_input.native_counterterm_trees(program) is None


def _program_flag(on):
    """``dag_input.NATIVE_PROGRAM`` set to ``on``."""

    return dag_input.native_flags(program=on)


class _Delegating:
    """A model the native tables do not cover (not a ``CompiledUFOModel``)
    that answers every query like the one it wraps."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.loop_fast
def test_native_stage_falls_back_with_a_record(
    models, loop_sm_compiled_model, monkeypatch
):
    """Rule 2: a listed feature the native builder does not cover takes the
    oracle stage, which the blocks record (``dag_fallback``: the feature's key
    and the refusal) and ``fallback_record`` collects, in
    ``build_tree_blocks`` and ``CountertermBlocks.build``; a refusal of no
    listed feature is raised (fail closed)."""

    from types import SimpleNamespace

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    python = _tree_blocks(model, process, {"QCD": 5}, False)
    wrapped = _tree_blocks(_Delegating(model), process, {"QCD": 5}, True)
    assert wrapped.dag_fallback.startswith("model-type: model type _Delegating")
    assert not isinstance(wrapped.blocks_dag, dag_input.NativeDag)
    assert wrapped.blocks == python.blocks
    program = _python_program(
        loop_sm_compiled_model, "g g > g g g", {"QCD": 3}, {"QCD": 5}
    )
    keys = [item.key for item in dag_input.NATIVE_FALLBACKS]
    assert len(set(keys)) == len(keys)
    assert all(item.owner and item.follow_up for item in dag_input.NATIVE_FALLBACKS)
    for item in dag_input.NATIVE_FALLBACKS:

        def refuse(*args, feature=item, **kwargs):
            raise dag_input.NativeDagUnsupported(feature.feature, feature.key)

        monkeypatch.setattr(dag_input, "native_blocks", refuse)
        monkeypatch.setattr(dag_input, "native_counterterm_blocks", refuse)
        events: list = []
        fallen = _tree_blocks(
            model, process, {"QCD": 5}, True, progress_callback=events.append
        )
        assert fallen.dag_fallback == f"{item.key}: {item.feature}"
        assert fallen.blocks == python.blocks
        # The native stage's events: its start, then the fallback.
        assert [e for e in events if e["step"] == "native-dag"] == [
            {"step": "native-dag", "stage": "loop-blocks", "state": "start"},
            {
                "step": "native-dag",
                "stage": "loop-blocks",
                "state": "fallback",
                "feature": fallen.dag_fallback,
            },
        ]
    # The counterterm program takes the same path (one feature: the oracle
    # stage costs seconds).
    events = []
    with _flag(True):
        again = ct_closure.CountertermBlocks.build(
            program.recursion,
            program.blocks.blocks_dag.process,
            progress_callback=events.append,
        )
    last = dag_input.NATIVE_FALLBACKS[-1]
    assert again.blocks.dag_fallback == f"{last.key}: {last.feature}"
    assert again.blocks.blocks == program.blocks.blocks
    assert [e for e in events if e["step"] == "native-dag"] == [
        {"step": "native-dag", "stage": "counterterm-program", "state": "start"},
        {
            "step": "native-dag",
            "stage": "counterterm-program",
            "state": "fallback",
            "feature": again.blocks.dag_fallback,
        },
    ]
    record = dag_input.fallback_record(
        SimpleNamespace(
            blocks=fallen, counterterm_program=SimpleNamespace(blocks=again)
        )
    )
    assert record == {
        "loop_blocks": fallen.dag_fallback,
        "counterterm_program": again.blocks.dag_fallback,
    }
    # The generation record's entry (``generation_filters``).
    from pyamplicol.loops.v2.artifact import GeneratedProcess

    generated = GeneratedProcess(
        SimpleNamespace(
            slice=SimpleNamespace(
                blocks=fallen, counterterm_program=SimpleNamespace(blocks=again)
            )
        ),
        None,
        {},
        "",
    )
    assert generated.native_fallbacks() == record
    assert (
        dag_input.fallback_record(
            SimpleNamespace(blocks=python, counterterm_program=None)
        )
        == {}
    )

    # A refusal of no listed feature is raised at both switch sites.
    def unlisted(*args, **kwargs):
        raise dag_input.NativeDagUnsupported("an unlisted feature")

    monkeypatch.setattr(dag_input, "native_blocks", unlisted)
    monkeypatch.setattr(dag_input, "native_counterterm_blocks", unlisted)
    with pytest.raises(dag_input.NativeDagUnsupported, match="an unlisted feature"):
        _tree_blocks(model, process, {"QCD": 5}, True)
    with (
        _flag(True),
        pytest.raises(dag_input.NativeDagUnsupported, match="an unlisted feature"),
    ):
        ct_closure.CountertermBlocks.build(
            program.recursion, program.blocks.blocks_dag.process
        )
    with pytest.raises(ValueError, match="unknown native fallback"):
        dag_input.NativeDagUnsupported("x", "not-a-key")


def _perturbed_flows(model, monkeypatch, change):
    """``model.allowed_quantum_flows`` with ``change`` applied to every flow,
    on an empty per-model row cache (the rows are the model's answers)."""

    from dataclasses import replace

    original = model.allowed_quantum_flows

    def perturbed(*args, **kwargs):
        return tuple(
            replace(flow, **change(flow)) for flow in original(*args, **kwargs)
        )

    monkeypatch.setattr(model, "allowed_quantum_flows", perturbed)
    monkeypatch.setattr(dag_input, "_MODEL_ROWS", {})


def _mutated_tables(model, process, change):
    """The native builder on the input tables of ``process`` with ``change``
    applied (the refusals the Rust builder makes itself)."""

    import json

    tables = json.loads(
        dag_input.dag_input(process, model, max_coupling_orders={"QCD": 5})
    )
    change(tables)
    dag_input._call(dag_input._native().V2Dag.build, json.dumps(tables))


def _nan_bits():
    import struct

    return int(struct.unpack("<Q", struct.pack("<d", float("nan")))[0])


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    "key",
    (
        "model-type",
        "leg-pdg",
        "tuple-spin",
        "tuple-spin-source",
        "flavour-rule",
        "quantum-numbers",
        "nan-coupling",
        "nan-colour-weight",
        "colour-basis-key",
        "source-states",
        "reachability",
        "vertex-kind",
        "leg-label",
    ),
)
def test_native_tables_refuse_each_listed_feature(models, key, monkeypatch):
    """Rule 2: every listed feature is refused for real (``feature`` its
    key), by the input tables on the model's answers or the process, or by
    the Rust builder on the tables: u u~ > d d~ g with one answer changed."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.generation.dag_color import ColorEngine

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    orders = {"QCD": 5}
    build = lambda: dag_input.native_blocks(  # noqa: E731
        process, model, max_coupling_orders=orders
    )
    if key == "model-type":
        build = lambda: dag_input.native_blocks(  # noqa: E731
            process, _Delegating(model), max_coupling_orders=orders
        )
    elif key == "leg-pdg":
        legs = (replace(process.legs[0], outgoing_pdg=None), *process.legs[1:])
        process = replace(process, legs=legs)
    elif key == "tuple-spin":
        _perturbed_flows(model, monkeypatch, lambda flow: {"spin_state": (0, 1)})
    elif key == "tuple-spin-source":
        original = model._source_ir

        def source(pdg):
            # The final-state gluon's states (no crossing applied).
            found = original(pdg)
            if int(pdg) != 21:
                return found
            states = tuple(replace(state, spin_state=(0, 1)) for state in found.states)
            return replace(found, states=states)

        monkeypatch.setattr(model, "_source_ir", source)
    elif key == "flavour-rule":
        _perturbed_flows(model, monkeypatch, lambda flow: {"flavour_flow": (999,)})
    elif key == "quantum-numbers":
        _perturbed_flows(
            model,
            monkeypatch,
            lambda flow: {"quantum_number_flow": (("pyamplicol-test", "1"),)},
        )
    elif key == "nan-coupling":
        _perturbed_flows(
            model, monkeypatch, lambda flow: {"coupling": (float("nan"), 0.0)}
        )
    elif key == "nan-colour-weight":

        def nan_weight(tables):
            for vertex in tables["vertices"]:
                vertex["colour_weight"] = [_nan_bits(), 0]

        build = lambda: _mutated_tables(model, process, nan_weight)  # noqa: E731
    elif key == "colour-basis-key":
        original_states = ColorEngine.source_states_for_leg

        def keyed(self, leg):
            return tuple(
                replace(state, basis_key=("pyamplicol-test",))
                for state in original_states(self, leg)
            )

        monkeypatch.setattr(ColorEngine, "source_states_for_leg", keyed)
    elif key == "source-states":

        def many(tables):
            label, states, ids = tables["sources"][0]
            tables["sources"][0] = [label, states * 33, ids]

        build = lambda: _mutated_tables(model, process, many)  # noqa: E731
    elif key == "reachability":
        # A reachability rule of its own (every mask, every block current).
        other = SimpleNamespace(
            admits_mask=lambda mask: True, admits_block=lambda *args: True
        )
        build = lambda: dag_input.native_blocks(  # noqa: E731
            process, model, max_coupling_orders=orders, reachability=other
        )
    elif key == "vertex-kind":

        def repeated(tables):
            tables["vertices"].append(dict(tables["vertices"][0]))

        build = lambda: _mutated_tables(model, process, repeated)  # noqa: E731
    elif key == "leg-label":

        def labelled(tables):
            tables["legs"][0]["label"] = 65

        build = lambda: _mutated_tables(model, process, labelled)  # noqa: E731
    expected = {
        "tuple-spin-source": "tuple-spin",
        "nan-colour-weight": "nan-coupling",
    }.get(key, key)
    with pytest.raises(dag_input.NativeDagUnsupported) as refused:
        build()
    assert refused.value.feature == expected, str(refused.value)
    assert dag_input.listed_fallback(refused.value).startswith(f"{expected}: ")


@pytest.mark.loop_fast
def test_native_tables_refuse_negative_orders_in_limited_names(models, monkeypatch):
    """Fail closed (no listed fallback) for a reachable vertex with a
    negative coupling order in a limited name: the oracle bounds each
    current's accumulated orders, which a negative order can bring back
    within the limit, while the tables prune vertices by their own orders.
    An unlimited name is not pruned, so its negative orders build."""

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    original = model.vertex_coupling_orders

    def negative(vertex):
        orders = tuple(original(vertex))
        if 21 in {abs(int(p)) for p in vertex.particles}:
            return (*orders, ("NEG", -1))
        return orders

    monkeypatch.setattr(model, "vertex_coupling_orders", negative)
    monkeypatch.setattr(dag_input, "_MODEL_ROWS", {})
    with pytest.raises(dag_input.NativeDagUnsupported, match="NEG\\^-1") as refused:
        dag_input.native_blocks(
            process, model, max_coupling_orders={"QCD": 5, "NEG": 0}
        )
    assert refused.value.feature is None
    # The switch site raises it (rule 2: fail closed).
    with pytest.raises(dag_input.NativeDagUnsupported, match="NEG\\^-1"):
        _tree_blocks(model, process, {"QCD": 5, "NEG": 0}, True, born={"QCD": 3})
    dag_input.native_blocks(process, model, max_coupling_orders={"QCD": 5})


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("text", "orders", "born"),
    (
        # Photon and Z currents (shared evaluation contracts of the
        # electroweak bosons: the value-equivalence derivation).
        ("u u~ > e- e+ g g", {"QCD": 4, "QED": 2}, {"QCD": 2, "QED": 2}),
        ("g g > u u~ a g", {"QCD": 5, "QED": 1}, {"QCD": 3, "QED": 1}),
        ("e+ e- > u u~ g", {"QCD": 3, "QED": 2}, {"QCD": 1, "QED": 2}),
        # Tops with W_T != 0: the Ward and coloured-width queries name them
        # (amputated currents of all legs but one aside).
        ("g g > t t~ g", {"QCD": 5}, {"QCD": 3}),
    ),
)
def test_switched_tree_blocks_wider_controls(models, text, orders, born):
    compiled, model = models
    _python, native = _assert_switch_equal(compiled, model, text, orders, born=born)
    if "t" in text.split():
        from pyamplicol.loops.v2.artifact import coloured_widths

        dags = (native.blocks_dag, native.born_dag)
        assert {"t", "t~"} <= ward_breaking_fields(dags, compiled)
        assert set(coloured_widths(dags, compiled, {})) >= {"t"}


def _generation_digests(compiled, request, native_module, on):
    """Every stage digest of ``LoopSlice.generate`` (recursion route) and
    of its plan with native trees, with the native DAG stage ``on`` or off,
    and the decodes the generation and each export made (rule 9)."""

    from pyamplicol.loops.v2 import snapshot
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.loops.v2.plan_export import export_plan

    decodes = [dag_input.decode_count()]
    with _flag(on):
        slice_ = LoopSlice.generate(compiled, request, native_module)
        decodes.append(dag_input.decode_count())
        digests = snapshot.stage_digests(slice_)
        decodes.append(dag_input.decode_count())
        digests["S1'"] = snapshot.loop_blocks_digest(slice_.blocks)
        digests["S1''"] = snapshot.counterterm_program_digest(
            slice_.counterterm_program.blocks
        )
        digests["S2"] = snapshot.root_closure_digest(slice_)
        exported = export_plan(slice_, forward=False, trees=True, born_squared=True)
        decodes.append(dag_input.decode_count())
        digests["P"] = snapshot.programs_digest(exported.trees)
        digests["S8 trees"] = hashlib.sha256(exported.json().encode()).hexdigest()
    return slice_, digests, [b - a for a, b in itertools.pairwise(decodes)]


@pytest.mark.loop_physics
def test_switched_generation_is_output_identical_and_decodes_nothing(
    loop_sm_compiled_model, loop_native
):
    """A4 output identity on the production path: a recursion-route
    ``LoopSlice.generate`` with the native DAG stage gives the oracle
    stage's S1, S4, S5, S8 (plan bytes, also with native trees), S1', S1'',
    S2 and P digests. Rule 9: the generation and the plan export decode no
    native DAG; the native-trees export decodes exactly the remaining A5
    consumer of the consumer table (the loop-block DAG's tree rows,
    ``RUST-MIGRATION.md`` §3.A): the CT program's interaction program is
    native (``NATIVE_PROGRAM``)."""

    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        {"QCD": 5, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    _python, expected, python_decodes = _generation_digests(
        compiled, request, loop_native, False
    )
    switched, digests, decodes = _generation_digests(
        compiled, request, loop_native, True
    )
    assert python_decodes == [0, 0, 0]
    assert decodes == [0, 0, 1]
    assert isinstance(switched.blocks.blocks_dag, dag_input.NativeDag)
    program = switched.counterterm_program.blocks.blocks
    assert isinstance(program.blocks_dag, dag_input.NativeDag)
    assert switched.blocks.blocks_dag.decoded and not program.blocks_dag.decoded
    assert digests == expected
