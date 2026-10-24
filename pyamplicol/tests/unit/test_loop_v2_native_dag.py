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

import contextlib
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
    vetoed_particle_ids,
    ward_breaking_fields,
)

from . import test_dag_equivalence_contract_classes as contract_classes
from .test_loop_v2_ct_two_point import two_point_model  # noqa: F401 (fixture)

VARIANTS = ("transverse", "full")


@contextlib.contextmanager
def _flag(on):
    """``dag_input.NATIVE_DAG`` set to ``on`` (off: the oracle stages)."""

    before = dag_input.NATIVE_DAG
    dag_input.NATIVE_DAG = on
    try:
        yield
    finally:
        dag_input.NATIVE_DAG = before


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

    The child imports the extension, which starts a restricted Symbolica
    instance: like ``test_six_gluon_generation_completes_without_a_personal_key``
    it must not overlap another unlicensed Symbolica process of the user.
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
    expected["blocks"] = json.loads(
        json.dumps(loop_blocks._handle.blocks(loop_blocks._tables()))
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
    assert dag_input.u1_leaking_structures(native) == (
        ct_closure.u1_leaking_structures(program)
    )
    ((arguments, leaking),) = scans
    assert leaking
    (tmp_path / "u1.bin").write_bytes(native.dag.columns())
    request["u1"] = [str(tmp_path / "u1.bin"), arguments]
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
        "out['blocks'] = json.loads(json.dumps(handle.blocks(tables)))\n"
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
        "print(json.dumps(out))\n"
    )
    environment = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "request.json")],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
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


def _python_program(compiled_path, text, born, loop, veto=()):
    """The production Python CT program of a request, once per module."""

    from .test_loop_v2_ct_recursion import _program_and_rules

    key = (str(compiled_path), text, repr(born), repr(loop), tuple(veto))
    if key not in _PROGRAMS:
        with _flag(False):
            _PROGRAMS[key] = _program_and_rules(compiled_path, text, born, loop, veto)[
                0
            ]
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
    W_T != 0) and a top veto."""

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
    from pyamplicol.loops.v2.blocks import propagated_species

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


@pytest.mark.loop_fast
def test_switched_tree_blocks_equal_the_oracle_stage(models):
    """A4: ``build_tree_blocks`` takes the native builder under
    ``NATIVE_DAG`` (u u~ > d d~ g: quark lines, connection signs)."""

    compiled, model = models
    _assert_switch_equal(compiled, model, "u u~ > d d~ g", {"QCD": 5}, born={"QCD": 3})


@pytest.mark.loop_fast
def test_switched_counterterm_program_equals_the_oracle_stage(
    loop_sm_compiled_model, monkeypatch
):
    """A4: ``CountertermBlocks.build`` takes the native builder under
    ``NATIVE_DAG``: the S1'' digest (DAG, blocks, clone lines) equals the
    oracle stage's, the U(1) leakage query runs natively (equal with the
    projection gate off too) and nothing is decoded."""

    from pyamplicol.loops.v2 import snapshot

    from .test_loop_v2_ct_recursion import _program_and_rules

    text, born, loop = "u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}
    python = _python_program(loop_sm_compiled_model, text, born, loop)
    before = dag_input.decode_count()
    with _flag(True):
        native = _program_and_rules(loop_sm_compiled_model, text, born, loop)[0]
    assert isinstance(native.blocks.blocks_dag, dag_input.NativeDag)
    assert native.blocks.dag_fallback is None
    assert snapshot.counterterm_program_digest(native) == (
        snapshot.counterterm_program_digest(python)
    )
    assert native.blocks.blocks == python.blocks.blocks
    assert dict(native.clones) == dict(python.clones)
    monkeypatch.setattr(ct_closure, "u1_projected", lambda compiled: False)
    leaking = ct_closure.u1_leaking_structures(native)
    assert leaking
    assert leaking == ct_closure.u1_leaking_structures(python)
    assert dag_input.decode_count() == before


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
    """Rule 2: a feature the native builder does not cover takes the oracle
    stage, which the blocks record (``dag_fallback``): a model other than a
    compiled UFO model, and each listed refusal of the native tables, in
    ``build_tree_blocks`` and ``CountertermBlocks.build``."""

    compiled, model = models
    process = _process(compiled, "u u~ > d d~ g")
    python = _tree_blocks(model, process, {"QCD": 5}, False)
    wrapped = _tree_blocks(_Delegating(model), process, {"QCD": 5}, True)
    assert "model type" in wrapped.dag_fallback
    assert not isinstance(wrapped.blocks_dag, dag_input.NativeDag)
    assert wrapped.blocks == python.blocks
    program = _python_program(
        loop_sm_compiled_model, "u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}
    )
    for feature in dag_input.NATIVE_FALLBACKS:

        def refuse(*args, feature=feature, **kwargs):
            raise dag_input.NativeDagUnsupported(feature)

        monkeypatch.setattr(dag_input, "native_blocks", refuse)
        monkeypatch.setattr(dag_input, "native_counterterm_blocks", refuse)
        fallen = _tree_blocks(model, process, {"QCD": 5}, True)
        assert fallen.dag_fallback == feature
        assert fallen.blocks == python.blocks
    # The counterterm program takes the same path (one feature: the oracle
    # stage costs seconds).
    with _flag(True):
        again = ct_closure.CountertermBlocks.build(
            program.recursion, program.blocks.blocks_dag.process
        )
    assert again.blocks.dag_fallback == dag_input.NATIVE_FALLBACKS[-1]
    assert again.blocks.blocks == program.blocks.blocks


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
    native DAG; the native-trees export decodes exactly the two A5 consumers
    of the consumer table (the loop-block and CT DAGs' interaction
    programs, ``RUST-MIGRATION.md`` §3.A)."""

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
    assert decodes == [0, 0, 2]
    assert isinstance(switched.blocks.blocks_dag, dag_input.NativeDag)
    program = switched.counterterm_program.blocks.blocks
    assert isinstance(program.blocks_dag, dag_input.NativeDag)
    assert switched.blocks.blocks_dag.decoded and program.blocks_dag.decoded
    assert digests == expected
