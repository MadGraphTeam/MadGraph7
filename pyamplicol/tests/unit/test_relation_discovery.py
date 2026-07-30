# SPDX-License-Identifier: 0BSD
"""Numerical nomination and exact-certificate promotion contracts."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    ExactCurrentRelationCertificate,
    _current_evaluation_contract,
    apply_numerical_current_relation_certificates,
    assign_recursive_current_evaluation_reuse,
    certify_numerical_current_observations,
    discover_recursive_evaluation_relations,
    generic_dag_numerical_source_semantics_sha256,
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


def _authenticated_relation_for_ambiguous_dag(
    relation_kind: str,
):
    dag, model = _ambiguous_projective_dag()
    source_digest = generic_dag_numerical_source_semantics_sha256(
        dag,
        execution_mode="compiled",
    )
    representative_candidate = (
        (Decimal("1.25"), Decimal("-0.5")),
        (Decimal("-3.0"), Decimal("2.25")),
        (Decimal("0.75"), Decimal("4.0")),
        (Decimal("-2.5"), Decimal("-1.125")),
    )
    representative_verification = (
        (Decimal("5.5"), Decimal("0.25")),
        (Decimal("-1.75"), Decimal("-3.0")),
        (Decimal("2.125"), Decimal("6.25")),
        (Decimal("-4.0"), Decimal("1.5")),
    )
    if relation_kind == "zero":
        candidate = tuple((Decimal(0), Decimal(0)) for _ in range(4))
        verification = candidate
        candidate_representative = None
        verification_representative = None
        representative_id = None
    else:
        sign = Decimal(1) if relation_kind == "equal" else Decimal(-1)
        candidate = tuple(
            (sign * real, sign * imaginary)
            for real, imaginary in representative_candidate
        )
        verification = tuple(
            (sign * real, sign * imaginary)
            for real, imaginary in representative_verification
        )
        candidate_representative = representative_candidate
        verification_representative = representative_verification
        representative_id = 3
    certificate = certify_numerical_current_observations(
        current_id=5,
        representative_id=representative_id,
        relation_kind=relation_kind,  # type: ignore[arg-type]
        source_semantics_sha256=source_digest,
        candidate_current_values=candidate,
        candidate_representative_values=candidate_representative,
        verification_current_values=verification,
        verification_representative_values=verification_representative,
        precision_digits=96,
        seed=0x5059414D,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )
    assert certificate is not None
    return dag, model, certificate


@pytest.mark.parametrize("relation_kind", ("equal", "opposite", "zero"))
def test_authenticated_numerical_relations_are_applied_by_default(
    relation_kind: str,
) -> None:
    dag, model, certificate = _authenticated_relation_for_ambiguous_dag(
        relation_kind
    )

    result = apply_numerical_current_relation_certificates(
        dag,
        model,
        (certificate,),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.report.state == "authenticated-numerical-applied"
    assert result.report.certificate_replay_status == "verified"
    assert result.report.applied_relation_count == 1
    assert result.report.warning_required
    assert result.report.interaction_evaluation_count_before == 4
    assert result.report.interaction_evaluation_count_projected == 3
    payload = result.report.to_json_dict()
    assert payload["relation_kind_counts"][relation_kind] == 1
    assert payload["warning"] == {
        "required": True,
        "emit": "once-per-generated-artifact",
        "code": "proofless-numerical-current-relations-applied-v1",
        "message": (
            "applied authenticated numerical equal/opposite/zero current "
            "reuse without an exact structural proof; disable with "
            "--no-numerical-current-reuse"
        ),
    }
    mapping = payload["mappings"][0]
    assert mapping["current_id"] == 5
    assert mapping["relation_kind"] == relation_kind
    if relation_kind == "zero":
        assert mapping["representative_id"] is None
        assert mapping["execution_representative_id"] == 3
        assert mapping["factor_binary64"] == ["0x0.0p+0", "0x0.0p+0"]


def test_empty_numerical_audit_is_explicit_and_does_not_warn_or_apply() -> None:
    dag, model = _ambiguous_projective_dag()

    result = apply_numerical_current_relation_certificates(
        dag,
        model,
        (),
        mode="certified-reuse",
        execution_mode="compiled",
    )

    assert result.dag is dag
    assert result.report.state == "no_certified_numerical_relation"
    assert (
        result.report.certificate_replay_status
        == "no_certified_numerical_relation"
    )
    assert result.report.applied_relation_count == 0
    assert not result.report.warning_required
    assert result.report.to_json_dict()["warning"] == {
        "required": False,
        "emit": "never",
        "code": None,
        "message": None,
    }


def test_numerical_diagnostic_mode_records_but_does_not_apply_or_warn() -> None:
    dag, model, certificate = _authenticated_relation_for_ambiguous_dag("equal")

    result = apply_numerical_current_relation_certificates(
        dag,
        model,
        (certificate,),
        mode="diagnostic",
        execution_mode="compiled",
    )

    assert result.dag is dag
    assert result.report.state == "authenticated-numerical-diagnostic-only"
    assert result.report.applied_relation_count == 0
    assert not result.report.warning_required
    assert result.report.interaction_evaluation_count_projected == 3


def test_numerical_relation_set_fails_closed_on_mode_or_source_drift() -> None:
    dag, model, certificate = _authenticated_relation_for_ambiguous_dag("equal")

    with pytest.raises(ValueError, match="does not replay"):
        apply_numerical_current_relation_certificates(
            dag,
            model,
            (certificate,),
            mode="certified-reuse",
            execution_mode="eager",
        )

    drifted = replace(
        dag,
        interactions=(
            replace(dag.interactions[0], color_weight=(7.0, 0.0)),
            *dag.interactions[1:],
        ),
    )
    with pytest.raises(ValueError, match="does not replay"):
        apply_numerical_current_relation_certificates(
            drifted,
            model,
            (certificate,),
            mode="certified-reuse",
            execution_mode="compiled",
        )
