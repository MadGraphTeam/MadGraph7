# SPDX-License-Identifier: 0BSD
"""Numerical nomination and exact-certificate promotion contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    ExactCurrentRelationCertificate,
    _current_evaluation_contract,
    assign_recursive_current_evaluation_reuse,
    discover_recursive_evaluation_relations,
    verify_dag_relation_certificates,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _ambiguous_projective_dag():
    """Build a tiny DAG whose existing projective index safely fails closed."""

    model = BuiltinSMModel()
    seed = compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="lc"),
        model=model,
    )
    source_templates = []
    source_keys = set()
    for source_id in seed.sources:
        source = seed.currents[source_id]
        key = (
            _current_evaluation_contract(source),
            source.source_leg_label,
            source.source_helicity,
        )
        if key in source_keys:
            continue
        source_keys.add(key)
        source_templates.append(source)
        if len(source_templates) == 3:
            break
    assert len(source_templates) == 3
    sources = tuple(
        replace(source, id=current_id)
        for current_id, source in enumerate(source_templates)
    )

    template = seed.interactions[0]
    generated_template = seed.currents[template.result_id]
    generated = tuple(
        replace(generated_template, id=current_id) for current_id in range(3, 8)
    )

    def interaction(
        interaction_id: int,
        left_id: int,
        right_id: int,
        result_id: int,
        color_weight: tuple[float, float],
    ):
        return replace(
            template,
            id=interaction_id,
            left_id=left_id,
            right_id=right_id,
            result_id=result_id,
            color_weight=color_weight,
            evaluation_group_id=None,
            evaluation_factor=(1.0, 0.0),
        )

    interactions = (
        interaction(0, 0, 1, 3, (3.0, 0.0)),
        interaction(1, 0, 2, 3, (6.0, 0.0)),
        interaction(2, 0, 1, 4, (1.0, 0.0)),
        interaction(3, 0, 2, 4, (2.0, 0.0)),
        interaction(4, 0, 1, 5, (6.0, 0.0)),
        interaction(5, 0, 2, 5, (12.0, 0.0)),
        interaction(6, 3, 1, 6, (1.0, 0.0)),
        interaction(7, 5, 1, 7, (1.0, 0.0)),
    )
    root = replace(
        seed.amplitude_roots[0],
        id=0,
        left_id=7,
        right_id=1,
    )
    dag = replace(
        seed,
        currents=(*sources, *generated),
        sources=(0, 1, 2),
        interactions=interactions,
        amplitude_roots=(root,),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_topology_replay=None,
        helicity_recurrence=None,
        helicity_materialization=None,
    )
    return dag, model


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_exact_certified_discovery_promotes_missed_dag_reuse(
    execution_mode,
) -> None:
    dag, model = _ambiguous_projective_dag()
    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    assert (
        baseline.interactions[6].evaluation_group_id
        != baseline.interactions[7].evaluation_group_id
    )

    result = discover_recursive_evaluation_relations(
        dag,
        model,
        mode="certified-reuse",
        execution_mode=execution_mode,
        precision_digits=80,
        probe_count=4,
        seed=17,
    )

    report = result.report
    assert report.execution_mode == execution_mode
    assert report.color_accuracy == "lc"
    assert report.state == "exact-certified-applied"
    assert report.numerical_candidate_count >= 3
    assert report.uncertified_candidate_count >= 1
    assert report.exact_certified_relation_count == 1
    assert report.applied_relation_count == 1
    assert report.certificate_replay_status == "verified"
    assert (
        result.dag.interactions[6].evaluation_group_id
        == result.dag.interactions[7].evaluation_group_id
    )
    assert verify_dag_relation_certificates(dag, model, report.certificates)
    restored = ExactCurrentRelationCertificate.from_json_dict(
        report.certificates[0].to_json_dict()
    )
    assert restored == report.certificates[0]
    assert verify_dag_relation_certificates(dag, model, (restored,))

    tampered = replace(report.certificates[0], factor=(3.0, 0.0))
    assert not verify_dag_relation_certificates(dag, model, (tampered,))


def test_diagnostic_mode_reports_promotable_reuse_without_applying_it() -> None:
    dag, model = _ambiguous_projective_dag()

    result = discover_recursive_evaluation_relations(
        dag,
        model,
        mode="diagnostic",
        execution_mode="compiled",
        precision_digits=96,
        probe_count=3,
        seed=23,
    )

    assert result.dag is dag
    assert result.report.state == "diagnostic-only"
    assert result.report.applied_relation_count == 0
    assert result.report.exact_certified_relation_count == 1
    assert result.report.interaction_evaluation_count_after is not None
    assert result.report.interaction_evaluation_count_before is not None
    assert (
        result.report.interaction_evaluation_count_after
        < result.report.interaction_evaluation_count_before
    )
    payload = result.report.to_json_dict()
    assert payload["probe"] == {
        "status": "completed",
        "precision_digits": 96,
        "probe_count": 3,
        "seed": 23,
        "deterministic": True,
        "candidate_only": True,
    }
