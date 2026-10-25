# SPDX-License-Identifier: 0BSD
"""Reuse derivation restricted to the currents that share a contract.

``assign_recursive_current_evaluation_reuse`` derives term vectors only for
the currents whose evaluation contract another current shares; every other
current is its own representative (WP3.1 Rust migration, A0). The oracle is the
full derivation:

* an electroweak (photon/Z) tree DAG with colour-sector copies, whose classes
  merge with the factors ±1, ±i, 2, 4 and 8, is kept as a fixture with its
  ``_CurrentValueEquivalence`` list (the compiler gives every current of the
  production tree, loop-block and CT DAGs of the controls a contract of its
  own or an identity class, so the merges come from the copies);
* a mutation check: the shortcut forced onto one shared-contract class
  changes that list;
* the structural refusals of the full derivation hold for every current;
* the loop blocks and the CT program, whose closed colour components share
  contracts, of two QCD controls (``loop_physics``; six legs nightly), on
  the Python stages (``dag_input.NATIVE_DAG`` off), and the native stages'
  DAGs equal to them.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pyamplicol.generation import dag_equivalence
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    _CurrentValueEquivalence,
    _derive_current_value_equivalences,
    _rewrite_interaction_evaluation_reuse,
    _shared_contract_current_ids,
    assign_recursive_current_evaluation_reuse,
)
from pyamplicol.generation.dag_types import GenericDAG
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "tree"
    / "dag_equivalence_ew_sector_copies.json"
)
# The colour-sector copies of the electroweak DAG multiply every colour weight
# of copy k by FACTORS[k - 1], so a current with n generated ancestors (itself
# included) is FACTORS[k - 1] ** n times its original. One interaction of the
# last copy is deformed by one part in 2**40: its result and every current
# built on it keep their own class.
EW_PROCESS = "u u~ > e+ e- g"
FACTORS = ((1.0, 0.0), (-1.0, 0.0), (2.0, 0.0), (0.0, 1.0))
DEFORMATION = (1.0 + 2.0**-40, 0.0)


def _mul(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _sector_copies() -> tuple[GenericDAG, BuiltinSMModel]:
    model = BuiltinSMModel()
    base = compile_generic_dag(
        build_process_ir(EW_PROCESS, color_accuracy="lc"), model=model
    )
    count, interaction_count = len(base.currents), len(base.interactions)
    currents = list(base.currents)
    sources = list(base.sources)
    interactions = [
        replace(item, evaluation_group_id=None, evaluation_factor=(1.0, 0.0))
        for item in base.interactions
    ]
    for copy, factor in enumerate(FACTORS, start=1):
        shift = copy * count
        for current in base.currents:
            state = current.index.color_state
            index = replace(
                current.index,
                color_state=replace(state, sector_id=state.sector_id + 100 * copy),
            )
            currents.append(replace(current, id=current.id + shift, index=index))
            if current.is_source:
                sources.append(current.id + shift)
        for item in base.interactions:
            interactions.append(
                replace(
                    item,
                    id=item.id + copy * interaction_count,
                    left_id=item.left_id + shift,
                    right_id=item.right_id + shift,
                    result_id=item.result_id + shift,
                    color_weight=_mul(factor, item.color_weight),
                    evaluation_group_id=None,
                    evaluation_factor=(1.0, 0.0),
                )
            )
    # The first interaction of the last copy whose result another one reads.
    parents = {item.left_id for item in base.interactions} | {
        item.right_id for item in base.interactions
    }
    deformed = (
        next(item.id for item in base.interactions if item.result_id in parents)
        + len(FACTORS) * interaction_count
    )
    interactions[deformed] = replace(
        interactions[deformed],
        color_weight=_mul(DEFORMATION, interactions[deformed].color_weight),
    )
    dag = replace(
        base,
        currents=tuple(currents),
        sources=tuple(sources),
        interactions=tuple(interactions),
        lc_topology_replay=None,
        helicity_recurrence=None,
        helicity_materialization=None,
    )
    return dag, model


def _records(
    equivalences: tuple[_CurrentValueEquivalence, ...],
) -> list[list[object]]:
    return [
        [item.representative_id, item.factor[0].hex(), item.factor[1].hex()]
        for item in equivalences
    ]


def _full_reuse(dag: GenericDAG, model: BuiltinSMModel) -> GenericDAG:
    """The reuse pass with the full derivation (the oracle)."""

    by_kind: dict = {}
    return _rewrite_interaction_evaluation_reuse(
        dag,
        model,
        current_equivalences=_derive_current_value_equivalences(
            dag, model, equivalence_by_kind=by_kind
        ),
        equivalence_by_kind=by_kind,
    )


def test_electroweak_sector_copies_keep_their_merged_classes() -> None:
    dag, model = _sector_copies()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["process"] == EW_PROCESS
    assert (len(dag.currents), len(dag.interactions)) == (
        fixture["currents"],
        fixture["interactions"],
    )
    shared = _shared_contract_current_ids(dag)
    assert len(shared) == fixture["shared_contract_currents"]

    full = _derive_current_value_equivalences(dag, model)
    restricted = _derive_current_value_equivalences(
        dag, model, derived_current_ids=shared
    )
    assert _records(full) == fixture["equivalences"]
    assert restricted == full
    merged = [
        item.factor
        for index, item in enumerate(full)
        if item.representative_id != index
    ]
    assert len(merged) == fixture["merged"]
    assert {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (8.0, 0.0)} <= set(merged)
    # The deformed current and the currents built on it share contracts but
    # keep classes of their own.
    last_copy = len(FACTORS) * len(dag.currents) // (len(FACTORS) + 1)
    own = [
        index
        for index in sorted(shared)
        if index >= last_copy and full[index].representative_id == index
    ]
    assert len(own) == fixture["own_classes_in_last_copy"] > 1

    reused = assign_recursive_current_evaluation_reuse(dag, model)
    assert reused == _full_reuse(dag, model)
    assert reused.interaction_evaluation_count == fixture["evaluations"]


def test_forcing_the_shortcut_onto_one_shared_class_changes_the_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, model = _sector_copies()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    shared = _shared_contract_current_ids(dag)
    full = _derive_current_value_equivalences(dag, model)
    # One merged class: a generated current of copy 2 (factor -1) and its
    # representative, which share their contract.
    member = next(
        index
        for index, item in enumerate(full)
        if item.representative_id != index and item.factor == (-1.0, 0.0)
    )
    one_class = {
        index
        for index, item in enumerate(full)
        if item.representative_id == full[member].representative_id
    }
    assert len(one_class) > 1 and one_class <= shared

    mutated = _derive_current_value_equivalences(
        dag, model, derived_current_ids=shared - one_class
    )
    assert _records(mutated) != fixture["equivalences"]
    assert mutated[member] == _CurrentValueEquivalence(member, (1.0, 0.0))

    monkeypatch.setattr(
        dag_equivalence,
        "_shared_contract_current_ids",
        lambda graph: shared - one_class,
    )
    assert assign_recursive_current_evaluation_reuse(dag, model) != _full_reuse(
        dag, model
    )


@pytest.mark.parametrize(
    ("process", "accuracy"),
    (("g g > g g g", "nlc"), ("u u~ > d d~ a g", "lc")),
)
def test_tree_controls_with_contracts_of_their_own(process, accuracy) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir(process, color_accuracy=accuracy), model=model
    )
    assert not _shared_contract_current_ids(dag)
    assert assign_recursive_current_evaluation_reuse(dag, model) == _full_reuse(
        dag, model
    )


def test_restricted_derivation_keeps_the_structural_refusals() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="lc"), model=model
    )
    assert not _shared_contract_current_ids(dag)
    empty: frozenset[int] = frozenset()

    source = dag.sources[0]
    anonymous = replace(
        dag,
        currents=tuple(
            replace(current, source_leg_label=None) if current.id == source else current
            for current in dag.currents
        ),
    )
    for derived in (None, empty):
        with pytest.raises(ValueError, match="lacks physical source metadata"):
            _derive_current_value_equivalences(
                anonymous, model, derived_current_ids=derived
            )

    # A parent from a later subset: the root's left current reads the
    # largest generated current.
    largest = max(
        (current for current in dag.currents if not current.is_source),
        key=lambda current: (len(current.index.external_labels), current.id),
    )
    first = dag.interactions[0]
    assert len(dag.currents[first.result_id].index.external_labels) < len(
        largest.index.external_labels
    )
    misordered = replace(
        dag,
        interactions=(replace(first, left_id=largest.id), *dag.interactions[1:]),
    )
    for derived in (None, empty):
        with pytest.raises(ValueError, match="parents from an earlier subset"):
            _derive_current_value_equivalences(
                misordered, model, derived_current_ids=derived
            )

    with pytest.raises(ValueError, match="cannot discover or authenticate"):
        _derive_current_value_equivalences(
            dag, model, derived_current_ids=empty, authenticated_relations={}
        )


# ------------------------------------------------------- loop-request DAGs

LOOP_CONTROLS = {
    "gg_ggg": ("g g > g g g", 3),
    "uux_ggg": ("u u~ > g g g", 3),
    "gg_gggg": ("g g > g g g g", 4),
    "uux_gggg": ("u u~ > g g g g", 4),
}


@pytest.mark.parametrize(
    "name",
    (
        pytest.param("gg_ggg", marks=pytest.mark.loop_physics),
        pytest.param("uux_ggg", marks=pytest.mark.loop_physics),
        pytest.param("gg_gggg", marks=pytest.mark.loop_nightly),
        pytest.param("uux_gggg", marks=pytest.mark.loop_nightly),
    ),
)
def test_loop_blocks_and_counterterm_program_equal_the_full_derivation(
    name: str,
    loop_sm_compiled_model,
    loop_native,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every reuse pass of the slice's generation (Born, loop blocks, CT
    program) gives the full derivation's classes on the Python stages (the
    oracle: ``dag_input.NATIVE_DAG`` off); the native stages (WP3.1 A4, the
    production default) give the same loop-block and CT DAGs, evaluation
    groups and factors included (their S1'/S1'' digests)."""

    from pyamplicol.loops.v2 import dag_input, snapshot
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference import legacy_capture

    derive = dag_equivalence._derive_current_value_equivalences
    calls: list[tuple[int, int]] = []

    def checked(dag, model, **options):
        result = derive(dag, model, **options)
        derived = options.get("derived_current_ids")
        if derived is not None:
            assert result == derive(dag, model)
            calls.append((len(dag.currents), len(derived)))
        return result

    process, qcd = LOOP_CONTROLS[name]
    request = SliceRequest(
        process,
        {"QCD": qcd},
        {"QCD": qcd + 2},
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
    )
    with monkeypatch.context() as patch:
        patch.setattr(dag_equivalence, "_derive_current_value_equivalences", checked)
        patch.setattr(dag_input, "NATIVE_DAG", False)
        python = LoopSlice.generate(
            load_compiled_model(loop_sm_compiled_model), request, loop_native
        )
    # Born, loop blocks and CT program; the CT program's closed colour
    # components share contracts.
    assert len(calls) >= 3
    assert any(shared for _currents, shared in calls)
    monkeypatch.setattr(dag_input, "NATIVE_DAG", True)
    native = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model), request, loop_native
    )
    assert isinstance(native.blocks.blocks_dag, dag_input.NativeDag)
    assert snapshot.loop_blocks_digest(native.blocks) == (
        snapshot.loop_blocks_digest(python.blocks)
    )
    assert snapshot.counterterm_program_digest(
        native.counterterm_program.blocks
    ) == snapshot.counterterm_program_digest(python.counterterm_program.blocks)
