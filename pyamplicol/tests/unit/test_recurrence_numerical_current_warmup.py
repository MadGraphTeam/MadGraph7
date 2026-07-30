# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal

from pyamplicol.api.errors import ArtifactError
from pyamplicol.generation.recurrence_numerical_current_warmup import (
    capture_recurrence_current_observations,
    recurrence_numerical_current_opt_out_report,
    recurrence_numerical_source_semantics_sha256,
    run_recurrence_numerical_current_warmup,
    validate_recurrence_numerical_current_application,
)
from pyamplicol.generation.service import (
    _recurrence_relation_reporting,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.runtime.recurrence_exact._plan import (
    _RecurrenceExactPlan,
    _SourceTemplate,
)
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    _Contribution,
    _Current,
    _ExactFactor,
    _Executor,
    _MomentumForm,
    _MomentumTerm,
    _RecurrenceExactSectionsV1,
    _ReplayTarget,
    _RowGroup,
    _Source,
)


def _factor(value: int) -> _ExactFactor:
    return _ExactFactor(value, 1, 0, 1)


def _topology_replay_plan() -> _RecurrenceExactPlan:
    sections = _RecurrenceExactSectionsV1(
        process_id="synthetic_replay",
        strategy="topology-replay",
        semantic_digest="1" * 64,
        runtime_layout_digest="2" * 64,
        current_arena_components=4,
        amplitude_destination_count=1,
        parameter_value_count=0,
        external_source_count=2,
        currents=(
            _Current(0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, DIRECT_NONE_U32),
            _Current(1, 0, 0, 1, 1, 1, 0, 0, 0, 4, 1, DIRECT_NONE_U32),
            _Current(
                2,
                1,
                0,
                2,
                1,
                0,
                1,
                0,
                2,
                4,
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
            ),
            _Current(
                3,
                1,
                0,
                3,
                1,
                0,
                1,
                0,
                3,
                4,
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
            ),
        ),
        sources=(
            _Source(0, 0, 0, 0, 0, 0, 0),
            _Source(1, 1, 1, 0, 0, 0, 0),
        ),
        contributions=(
            _Contribution(
                0,
                DIRECT_NONE_U32,
                1,
                DIRECT_NONE_U32,
                2,
                0,
                0,
                3,
            ),
            _Contribution(
                0,
                DIRECT_NONE_U32,
                1,
                DIRECT_NONE_U32,
                3,
                0,
                0,
                3,
            ),
        ),
        finalizations=(),
        closures=(),
        row_groups=(
            _RowGroup(0, 0, 0, 0, 0, 2),
            _RowGroup(1, 1, 1, DIRECT_NONE_U32, 0, 2),
        ),
        momentum_forms=(_MomentumForm(0, 1), _MomentumForm(1, 1)),
        momentum_terms=(_MomentumTerm(0, 1), _MomentumTerm(1, 1)),
        replay_targets=(
            _ReplayTarget(10, 0, 0, 2, 0, 0, 0, 1, 0),
            _ReplayTarget(11, 0, 2, 2, 0, 0, 0, 1, 0),
        ),
        source_permutations=(0, 1, 1, 0),
        replay_momentum_signs=(1, 1, 1, 1),
        replay_helicity_map=(),
        amplitude_destinations=(),
        resolved_helicities=(),
        source_state_assignments=(),
        source_dispatch_variants=(),
        source_embeddings=(),
        source_projections=(),
        resolved_source_selections=(),
        public_helicities=(),
        exact_factors=(_factor(1),),
        public_flow_ids=(10, 11),
        executors=(
            _Executor(
                0,
                "source",
                "initialize",
                (),
                1,
                1,
                None,
                "source",
            ),
        ),
    )
    return _RecurrenceExactPlan(
        sections=sections,
        kernels={},
        executors={0: sections.executors[0]},
        executor_exact_kernel_ids={},
        executor_parent_permutations={0: (0, 1)},
        source_templates={
            0: _SourceTemplate(
                0,
                1,
                0,
                0,
                0,
                "scalar",
                "self-conjugate",
                None,
                1,
                1,
                1,
            )
        },
        initial_source_slots=frozenset({0, 1}),
        executor_couplings={},
        prepared_defaults=(),
        parameter_projection=(),
        parameter_derivation=None,
        runtime_defaults=(Decimal("1.25"), Decimal("2.5")),
    )


def _points(seed_start: int) -> tuple[ValidationPointRecord, ...]:
    return tuple(
        ValidationPointRecord(
            process_id="synthetic_replay",
            process="s s > s s",
            seed=seed,
            particles=(
                (1, (10.0 + seed, 0.0, 0.0, 10.0 + seed)),
                (-1, (20.0 + seed, 0.0, 0.0, -20.0 - seed)),
            ),
        )
        for seed in range(seed_start, seed_start + 2)
    )


def test_replay_capture_observes_contribution_only_currents() -> None:
    plan = _topology_replay_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    captured = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=41,
        domain="candidate-current-probes-v1",
    )

    assert captured.current_count == 4
    assert captured.current_dimensions == {0: 1, 1: 1, 2: 1, 3: 1}
    assert captured.observations[2] == captured.observations[3]
    assert captured.context_policy == "seeded-replay-target-per-physical-point-v1"
    assert len(captured.context_sha256s) == 2


def test_recurrence_certified_reuse_uses_two_independent_probe_sets() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert result.applied_relation_count == 1
    certificate = result.certificates[0]
    assert certificate.current_id == 3
    assert certificate.representative_id == 2
    assert certificate.execution_representative_id == 2
    assert certificate.relation_kind == "equal"
    assert certificate.current_dimension == 1
    assert certificate.candidate_probe_count == 2
    assert certificate.verification_probe_count == 2
    assert set(result.candidate_capture.kinematic_sha256s).isdisjoint(
        result.verification_capture.kinematic_sha256s
    )
    evidence = json.loads(result.evidence_json)
    assert evidence["requested_mode"] == "certified-reuse"
    assert evidence["mappings"][0]["factor_integer"] == [1, 0]
    assert result.to_json_dict()["warning"]["required"] is True

    validated = validate_recurrence_numerical_current_application(
        result,
        _topology_replay_plan(),
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    assert validated.application_validation["status"] == "verified"
    assert validated.application_validation["maximum_absolute_residual"] == "0"


def test_recurrence_diagnostic_certifies_without_applying_or_warning() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="diagnostic",
        color_accuracy="lc",
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert len(result.certificates) == 1
    assert result.applied_relation_count == 0
    report = result.to_json_dict()
    assert report["state"] == "authenticated-numerical-diagnostic-only"
    assert report["warning"]["required"] is False
    assert json.loads(result.evidence_json)["requested_mode"] == "diagnostic"


def test_recurrence_opt_out_has_no_capture_or_warning() -> None:
    report = recurrence_numerical_current_opt_out_report(
        _topology_replay_plan().sections,
        color_accuracy="lc",
    )

    assert report["requested_mode"] == "off"
    assert report["state"] == "disabled-by-user"
    assert report["candidate_capture"] is None
    assert report["verification_capture"] is None
    assert report["applied_relation_count"] == 0
    assert report["warning"]["required"] is False


def test_recurrence_aggregate_reports_applied_warning_and_explicit_opt_out() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    runtime_inspection, aggregate = _recurrence_relation_reporting(
        {"relation_discovery": {"state": "native-applied"}},
        mode="certified-reuse",
        lane_report=result.to_json_dict(),
    )
    assert "relation_discovery" not in runtime_inspection
    assert aggregate["execution_mode"] == "recurrence"
    assert aggregate["lanes"]["primary"]["applied_relation_count"] == 1
    assert aggregate["certified_relation_count"] == 1
    assert aggregate["applied_relation_count"] == 1
    assert aggregate["warning"]["required"] is True
    assert aggregate["warning"]["emit"] == "once-per-generated-artifact"
    assert aggregate["native_relation_application"] == {
        "state": "native-applied"
    }

    off_lane = recurrence_numerical_current_opt_out_report(
        _topology_replay_plan().sections,
        color_accuracy="lc",
    )
    off_runtime_inspection, off_aggregate = _recurrence_relation_reporting(
        {},
        mode="off",
        lane_report=off_lane,
    )
    assert "relation_discovery" not in off_runtime_inspection
    assert off_aggregate["requested_mode"] == "off"
    assert off_aggregate["lanes"]["primary"]["state"] == "disabled-by-user"
    assert off_aggregate["warning"]["required"] is False


def test_application_rejects_current_dimension_drift() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=79,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    drifted = _topology_replay_plan()
    current = drifted.sections.currents[3]
    drifted.sections = replace(
        drifted.sections,
        currents=(
            *drifted.sections.currents[:3],
            replace(current, component_count=2),
        ),
    )

    try:
        validate_recurrence_numerical_current_application(
            result,
            drifted,
            precision_digits=80,
            seed=79,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )
    except (ArtifactError, ValueError):
        pass
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("current-dimension drift was accepted")
