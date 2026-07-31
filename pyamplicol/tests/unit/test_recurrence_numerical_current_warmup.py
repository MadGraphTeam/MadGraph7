# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.generation.recurrence_numerical_current_warmup as recurrence_warmup
import pyamplicol.generation.service as generation_service
from pyamplicol.api.errors import ArtifactError, GenerationError
from pyamplicol.generation.dag_equivalence import (
    _advance_rejection_digest,
    _rejected_candidate_diagnostics,
)
from pyamplicol.generation.numerical_candidate_index import (
    build_numerical_observation_candidate_index,
    numerical_observation_tolerance_window_ids,
)
from pyamplicol.generation.recurrence_numerical_current_warmup import (
    _COMPRESSED_EVIDENCE_HEADER,
    _MAX_PERSISTED_EVIDENCE_BYTES,
    _MAX_RAW_EVIDENCE_BYTES,
    _MAX_RAW_EVIDENCE_MEMORY_BYTES,
    _MIN_RAW_EVIDENCE_WIRE_BYTES,
    _build_candidate_indexes,
    _decimal_string,
    _discover_relations,
    _pair_residuals,
    _raw_evidence_memory_upper_bound,
    _raw_evidence_wire_byte_limit,
    _raw_streaming_consumer_memory_upper_bound,
    _read_compressed_evidence_spool,
    _relation_residuals,
    _runtime_parameter_schema_payload,
    _spooled_capture_memory_upper_bound,
    _SpooledObservationMapping,
    _synthetic_raw_evidence_bytes,
    _synthetic_raw_evidence_bytes_by_dimension_counts,
    _validate_raw_evidence_canonical_size,
    _validate_raw_evidence_geometry,
    _write_compressed_canonical_evidence,
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
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.runtime.recurrence_exact._plan import (
    _RecurrenceExactPlan,
    _RuntimeParameterSchemaRow,
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


def test_opposite_residual_is_independent_of_ambient_decimal_context() -> None:
    exact = Decimal("0.123456789012345678901234567890123456789012345678901234567890")
    with localcontext() as context:
        context.prec = 7
        residuals = _pair_residuals(
            ((Decimal(f"-{exact}"), Decimal(0)),),
            ((exact, Decimal(0)),),
            sign=-1,
            relative_tolerance=Decimal("1e-70"),
            absolute_tolerance=Decimal("1e-80"),
        )

    assert residuals == (0, 0, 0)


@pytest.mark.parametrize(
    ("process_expression", "process_id"),
    (
        ("d d~ > z", "d_dbar_to_z"),
        ("u d~ > w+", "u_dbar_to_w"),
    ),
)
def test_massive_one_body_recurrence_probe_domains_are_physical_and_distinct(
    process_expression: str,
    process_id: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir(process_expression, color_accuracy="full")
    probes = recurrence_warmup.build_recurrence_numerical_current_probe_points(
        process,
        model,
        process_id=process_id,
        seed=17,
        candidate_count=4,
        verification_count=4,
    )

    assert probes == (
        recurrence_warmup.build_recurrence_numerical_current_probe_points(
            process,
            model,
            process_id=process_id,
            seed=17,
            candidate_count=4,
            verification_count=4,
        )
    )
    candidate, verification = probes
    candidate_hashes = {
        recurrence_warmup._kinematic_sha256(point) for point in candidate
    }
    verification_hashes = {
        recurrence_warmup._kinematic_sha256(point) for point in verification
    }
    assert len(candidate_hashes) == len(candidate)
    assert len(verification_hashes) == len(verification)
    assert candidate_hashes.isdisjoint(verification_hashes)

    for point in (*candidate, *verification):
        assert point.available
        assert point.process == process_expression
        assert point.process_id == process_id
        assert all(
            math.isfinite(component)
            for _pdg, momentum in point.particles
            for component in momentum
        )
        incoming = point.four_vectors[:2]
        outgoing = point.four_vectors[2:]
        for component in range(4):
            assert math.isclose(
                sum(momentum[component] for momentum in incoming),
                sum(momentum[component] for momentum in outgoing),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        for pdg, (energy, x_component, y_component, z_component) in point.particles:
            invariant_mass_squared = (
                energy * energy
                - x_component * x_component
                - y_component * y_component
                - z_component * z_component
            )
            expected_mass_squared = float(model.mass(pdg)) ** 2
            assert math.isclose(
                invariant_mass_squared,
                expected_mass_squared,
                rel_tol=1.0e-12,
                abs_tol=1.0e-10,
            )


def test_zero_residual_specialization_matches_explicit_zero_comparison() -> None:
    current = (
        (Decimal("0"), Decimal("-0")),
        (Decimal("1.25"), Decimal("-3.5")),
        (
            Decimal("0.123456789012345678901234567890123456789"),
            Decimal("-9.87654321098765432109876543210987654321"),
        ),
    )
    relative = Fraction(1, 10**70)
    absolute = Fraction(1, 10**80)

    expected = _pair_residuals(
        current,
        tuple((Decimal(0), Decimal(0)) for _ in current),
        sign=1,
        relative_tolerance=Decimal("1e-70"),
        absolute_tolerance=Decimal("1e-80"),
    )
    actual = _relation_residuals(
        "zero",
        current,
        None,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
    )

    assert actual == expected
    with pytest.raises(ValueError, match="relation width is invalid"):
        _relation_residuals(
            "equal",
            current,
            None,
            relative_tolerance=relative,
            absolute_tolerance=absolute,
        )


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
        runtime_parameter_schema=(
            _RuntimeParameterSchemaRow(0, "probe_a", 0, None, 0),
            _RuntimeParameterSchemaRow(1, "probe_b", 1, None, 0),
        ),
    )


def _no_relation_plan() -> _RecurrenceExactPlan:
    plan = _topology_replay_plan()
    contributions = list(plan.sections.contributions)
    contributions[1] = replace(contributions[1], exact_factor_id=1)
    plan.sections = replace(
        plan.sections,
        contributions=tuple(contributions),
        exact_factors=(*plan.sections.exact_factors, _factor(2)),
    )
    return plan


def _signed_relation_plan() -> _RecurrenceExactPlan:
    plan = _topology_replay_plan()
    prototype = plan.sections.currents[3]
    contribution = plan.sections.contributions[0]
    plan.sections = replace(
        plan.sections,
        current_arena_components=6,
        currents=(
            *plan.sections.currents,
            replace(
                prototype,
                semantic_id=4,
                component_base=4,
                first_use=4,
                last_use=4,
            ),
            replace(
                prototype,
                semantic_id=5,
                component_base=5,
                first_use=5,
                last_use=5,
            ),
        ),
        contributions=(
            *plan.sections.contributions,
            replace(
                contribution,
                destination_base=4,
                exact_factor_id=1,
            ),
            replace(
                contribution,
                destination_base=5,
                exact_factor_id=2,
            ),
        ),
        row_groups=(
            plan.sections.row_groups[0],
            replace(plan.sections.row_groups[1], row_count=4),
        ),
        exact_factors=(
            *plan.sections.exact_factors,
            _factor(-1),
            _factor(0),
        ),
    )
    return plan


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


def test_python_parameter_default_hex_matches_native_signed_binary64_contract() -> None:
    plan = _topology_replay_plan()
    plan.runtime_defaults = (
        Decimal.from_float(-1.5),
        Decimal.from_float(float.fromhex("-0x0.0000000000001p-1022")),
    )

    payload = _runtime_parameter_schema_payload(plan)

    assert [row["default_binary64"] for row in payload["parameters"]] == [
        "-0x1.8000000000000p+0",
        "-0x0.0000000000001p-1022",
    ]


def test_warmup_builds_static_semantics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_calls = 0
    default_calls = 0
    source_semantics_calls = 0
    original_contracts = recurrence_warmup._current_contracts
    original_defaults = recurrence_warmup._runtime_parameter_defaults
    original_source_semantics = recurrence_warmup._source_semantics_payload

    def counted_contracts(
        sections: _RecurrenceExactSectionsV1,
    ) -> tuple[tuple[object, ...], ...]:
        nonlocal contract_calls
        contract_calls += 1
        return original_contracts(sections)

    def counted_defaults(plan: _RecurrenceExactPlan) -> tuple[Decimal, ...]:
        nonlocal default_calls
        default_calls += 1
        return original_defaults(plan)

    def counted_source_semantics(
        sections: _RecurrenceExactSectionsV1,
        *,
        contracts: tuple[tuple[object, ...], ...] | None = None,
    ) -> dict[str, object]:
        nonlocal source_semantics_calls
        source_semantics_calls += 1
        return original_source_semantics(sections, contracts=contracts)

    monkeypatch.setattr(recurrence_warmup, "_current_contracts", counted_contracts)
    monkeypatch.setattr(
        recurrence_warmup,
        "_runtime_parameter_defaults",
        counted_defaults,
    )
    monkeypatch.setattr(
        recurrence_warmup,
        "_source_semantics_payload",
        counted_source_semantics,
    )
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
    try:
        assert contract_calls == 1
        assert default_calls == 1
        assert source_semantics_calls == 1
        candidate_dimensions = result.candidate_capture.current_dimensions
        verification_dimensions = result.verification_capture.current_dimensions
        assert candidate_dimensions == verification_dimensions
        assert candidate_dimensions is not verification_dimensions
        assert isinstance(candidate_dimensions, dict)
        assert isinstance(verification_dimensions, dict)
        current_id = next(iter(candidate_dimensions))
        verification_dimension = verification_dimensions[current_id]
        candidate_dimensions[current_id] = verification_dimension + 1
        assert verification_dimensions[current_id] == verification_dimension
    finally:
        result.close()


def test_warmup_phase_timings_are_diagnostic_only() -> None:
    arguments = {
        "candidate_points": _points(1),
        "verification_points": _points(101),
        "mode": "certified-reuse",
        "color_accuracy": "lc",
        "precision_digits": 80,
        "seed": 53,
        "relative_tolerance": 1.0e-60,
        "absolute_tolerance": 1.0e-70,
    }
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        **arguments,
    )
    repeated = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        **arguments,
    )
    try:
        assert result.generation_profile_timings
        assert all(
            isinstance(seconds, float) and seconds >= 0.0
            for seconds in result.generation_profile_timings.values()
        )
        assert {
            key
            for key in result.generation_profile_timings
            if key.startswith("warmup_candidate_probe_")
        } == {"warmup_candidate_probe_0", "warmup_candidate_probe_1"}
        assert {
            key
            for key in result.generation_profile_timings
            if key.startswith("warmup_verification_probe_")
        } == {"warmup_verification_probe_0", "warmup_verification_probe_1"}
        assert "generation_profile_timings" not in json.dumps(
            result.to_json_dict(),
            sort_keys=True,
        )
        assert result.evidence_json == repeated.evidence_json
        assert result == repeated
        assert result == replace(
            result,
            generation_profile_timings={"diagnostic-only": 1.0},
        )
        with pytest.raises(TypeError):
            result.generation_profile_timings["not-mutable"] = 1.0  # type: ignore[index]
    finally:
        result.close()
        repeated.close()


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
    assert result.candidate_capture.current_count == 4
    assert result.verification_capture.current_count == 4
    assert set(result.candidate_capture.observations) == {2, 3}
    assert set(result.verification_capture.observations) == {2, 3}
    assert result.candidate_capture.complete_observations_retained is False
    assert result.verification_capture.complete_observations_retained is False
    with pytest.raises(ValueError, match="cannot recreate raw evidence"):
        result.candidate_capture.to_evidence_dict()
    assert set(result.candidate_capture.kinematic_sha256s).isdisjoint(
        result.verification_capture.kinematic_sha256s
    )
    evidence = json.loads(result.evidence_json)
    assert evidence["requested_mode"] == "certified-reuse"
    assert "relative_tolerance" not in evidence
    assert "absolute_tolerance" not in evidence
    assert evidence["relative_tolerance_binary64"] == (1.0e-60).hex()
    assert evidence["absolute_tolerance_binary64"] == (1.0e-70).hex()
    assert evidence["mappings"][0]["factor_integer"] == [1, 0]
    assert len(evidence["candidate_capture"]["observations"]) == 4
    assert len(evidence["verification_capture"]["observations"]) == 4
    report = result.to_json_dict()
    assert report["warning"]["required"] is True
    rejected = report["discovery"]["rejected_candidate_diagnostics"]
    assert rejected["retained_count"] == len(report["discovery"]["rejected_candidates"])
    assert rejected["total_rejected_hypothesis_count"] >= rejected["retained_count"]
    assert rejected["full_decision_sha256"] == report["discovery"]["decision_sha256"]
    persisted = report["persisted_numerical_evidence"]
    assert persisted["raw_evidence_retained"] is False
    census = persisted["full_census"]
    assert (
        census["uncertified_candidate_count"] == census["verification_rejected_count"]
    )
    assert census["numerical_candidate_count"] == (
        census["certified_relation_count"] + census["uncertified_candidate_count"]
    )
    assert census["rejected_hypothesis_count"] == (
        census["tested_hypothesis_count"] - census["certified_relation_count"]
    )
    assert (
        census["rejection_decision_sha256"]
        == report["discovery"]["rejection_decision_sha256"]
    )
    assert persisted["measured_payload_bytes"] < _MAX_PERSISTED_EVIDENCE_BYTES
    assert {
        row["current_id"] for row in persisted["candidate_capture"]["observations"]
    } == {2, 3}
    assert persisted["candidate_capture"]["full_batch_commitment"]["current_count"] == 4
    assert len(json.dumps(persisted, separators=(",", ":"), sort_keys=True)) < len(
        result.evidence_json
    )

    validated = validate_recurrence_numerical_current_application(
        result,
        _topology_replay_plan(),
        baseline_plan=_topology_replay_plan(),
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
    assert result.candidate_capture.current_count == 4
    assert result.verification_capture.current_count == 4
    assert result.candidate_capture.observations == {}
    assert result.verification_capture.observations == {}
    assert result.candidate_capture.complete_observations_retained is False
    assert result.verification_capture.complete_observations_retained is False
    report = result.to_json_dict()
    assert report["state"] == "authenticated-numerical-diagnostic-only"
    assert report["warning"]["required"] is False
    assert json.loads(result.evidence_json)["requested_mode"] == "diagnostic"


def test_recurrence_no_relation_detaches_all_observations_with_full_census() -> None:
    result = run_recurrence_numerical_current_warmup(
        _no_relation_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    assert result.certificates == ()
    assert result.candidate_capture.current_count == 4
    assert result.verification_capture.current_count == 4
    assert result.candidate_capture.observations == {}
    assert result.verification_capture.observations == {}
    assert result.discovery_report["inspected_current_count"] == 4
    assert result.discovery_report["tested_hypothesis_count"] > 0
    evidence = json.loads(result.evidence_json)
    assert len(evidence["candidate_capture"]["observations"]) == 4
    assert len(evidence["verification_capture"]["observations"]) == 4


def test_application_recapture_rejects_stale_plan_and_capture_commitment() -> None:
    result = run_recurrence_numerical_current_warmup(
        _topology_replay_plan(),
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=73,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )

    with pytest.raises(ValueError, match="did not reproduce"):
        validate_recurrence_numerical_current_application(
            result,
            _topology_replay_plan(),
            baseline_plan=_no_relation_plan(),
            precision_digits=80,
            seed=73,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )

    tampered = replace(
        result,
        verification_capture=replace(
            result.verification_capture,
            observation_batch_sha256="0" * 64,
        ),
    )
    with pytest.raises(ValueError, match="did not reproduce"):
        validate_recurrence_numerical_current_application(
            tampered,
            _topology_replay_plan(),
            baseline_plan=_topology_replay_plan(),
            precision_digits=80,
            seed=73,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )


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


def test_raw_evidence_scaling_is_quantified_and_bounded() -> None:
    common = {
        "component_count": 4,
        "candidate_probe_count": 4,
        "verification_probe_count": 4,
        "decimal_characters": 112,
    }
    b_like_bytes = _synthetic_raw_evidence_bytes(652, **common)
    a_like_bytes = _synthetic_raw_evidence_bytes(17_000, **common)

    assert b_like_bytes == 4_918_550
    assert a_like_bytes == 128_293_862
    assert a_like_bytes < _MAX_RAW_EVIDENCE_BYTES
    assert a_like_bytes > 25 * b_like_bytes
    a_like_upper_bound = _raw_evidence_memory_upper_bound(
        a_like_bytes,
        scalar_count=1_088_000,
        row_count=34_000,
    )
    assert a_like_upper_bound == 1_003_870_156
    assert a_like_upper_bound < _MAX_RAW_EVIDENCE_MEMORY_BYTES
    just_over = _synthetic_raw_evidence_bytes(50_000, **common)
    assert just_over > _MAX_RAW_EVIDENCE_BYTES
    with pytest.raises(ValueError, match="memory envelope"):
        _validate_raw_evidence_canonical_size(just_over)


def test_real_a_mixed_dimension_shape_uses_dynamic_wire_budget() -> None:
    dimension_counts = {4: 15_834, 6: 1_240}
    current_count = sum(dimension_counts.values())
    component_count = sum(
        dimension * count for dimension, count in dimension_counts.items()
    )
    point_count = 4 + 4
    runtime_parameter_count = 10
    scalar_count = (
        2 * component_count * point_count + runtime_parameter_count * point_count
    )
    row_count = current_count * 2 + runtime_parameter_count
    byte_limit = _raw_evidence_wire_byte_limit(
        scalar_count=scalar_count,
        row_count=row_count,
    )

    assert (current_count, component_count) == (17_074, 70_776)
    assert (scalar_count, row_count) == (1_132_496, 34_158)
    assert byte_limit == 148_950_528
    wire_sizes = {
        characters: _synthetic_raw_evidence_bytes_by_dimension_counts(
            dimension_counts,
            candidate_probe_count=4,
            verification_probe_count=4,
            decimal_characters=characters,
        )
        for characters in (96, 112, 128)
    }
    assert wire_sizes == {
        96: 115_356_478,
        112: 133_475_134,
        128: 151_593_790,
    }
    _validate_raw_evidence_canonical_size(
        wire_sizes[96],
        byte_limit=byte_limit,
    )
    _validate_raw_evidence_canonical_size(
        wire_sizes[112],
        byte_limit=byte_limit,
    )
    with pytest.raises(ValueError, match="memory envelope"):
        _validate_raw_evidence_canonical_size(
            wire_sizes[128],
            byte_limit=byte_limit,
        )


def test_real_a_streaming_consumer_bound_matches_native_model() -> None:
    resident = _raw_streaming_consumer_memory_upper_bound(
        raw_byte_count=146_798_789,
        metadata_byte_count=2_874_885,
        metadata_structural_token_count=788_978,
        current_count=17_074,
        component_count=70_776,
        maximum_dimension=6,
        candidate_probe_count=4,
        verification_probe_count=4,
        runtime_parameter_count=10,
    )

    assert resident == 414_500_724
    assert resident < _MAX_RAW_EVIDENCE_MEMORY_BYTES


def test_runtime_parameter_metadata_reduces_dynamic_wire_budget() -> None:
    without_parameters = _raw_evidence_wire_byte_limit(
        scalar_count=1_132_416,
        row_count=34_148,
    )
    with_ten_parameters = _raw_evidence_wire_byte_limit(
        scalar_count=1_132_496,
        row_count=34_158,
    )

    assert without_parameters == 148_978_688
    assert with_ten_parameters == 148_950_528
    assert without_parameters - with_ten_parameters == 28_160


def test_parameter_context_geometry_is_preflighted_without_allocation() -> None:
    sections = _topology_replay_plan().sections
    with pytest.raises(ValueError, match="memory envelope"):
        _validate_raw_evidence_geometry(
            sections,
            candidate_probe_count=2,
            verification_probe_count=2,
            runtime_parameter_count=400_000,
        )


def test_exact_z_n8_sequential_spool_bound_includes_the_global_index() -> None:
    resident = _spooled_capture_memory_upper_bound(
        current_count=38_581,
        component_count=162_976,
        maximum_probe_count=4,
        runtime_parameter_count=10,
    )

    assert resident == 935_671_296
    assert resident < _MAX_RAW_EVIDENCE_MEMORY_BYTES
    assert _MAX_RAW_EVIDENCE_MEMORY_BYTES - resident == 138_070_528


def test_compressed_canonical_transport_round_trips_with_length_and_digest() -> None:
    value = {
        "candidate_capture": {"observations": []},
        "verification_capture": {"observations": []},
    }
    expected = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    with tempfile.TemporaryFile() as stream:
        canonical_bytes = _write_compressed_canonical_evidence(
            value,
            stream,
            canonical_byte_limit=1 << 20,
        )
        encoded = _read_compressed_evidence_spool(stream)

    magic, declared_bytes, digest = _COMPRESSED_EVIDENCE_HEADER.unpack_from(encoded)
    assert magic == b"PACNCEZ1"
    assert canonical_bytes == declared_bytes == len(expected)
    assert zlib.decompress(encoded[_COMPRESSED_EVIDENCE_HEADER.size :]) == expected
    assert digest.hex() == hashlib.sha256(expected).hexdigest()


def test_compressed_warmup_keeps_relation_rows_spooled_through_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recurrence_warmup,
        "_select_raw_evidence_storage_geometry",
        lambda *_args, **_kwargs: recurrence_warmup._RawEvidenceStorageGeometry(
            scalar_count=128,
            row_count=16,
            canonical_byte_limit=1 << 20,
            encoding="zlib-canonical-json-v1",
            producer_resident_upper_bound=1 << 20,
        ),
    )
    plan = _topology_replay_plan()
    result = run_recurrence_numerical_current_warmup(
        plan,
        candidate_points=_points(1),
        verification_points=_points(101),
        mode="certified-reuse",
        color_accuracy="lc",
        precision_digits=80,
        seed=53,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
    )
    candidate_spool = result.candidate_capture.observations
    verification_spool = result.verification_capture.observations
    assert isinstance(candidate_spool, _SpooledObservationMapping)
    assert isinstance(verification_spool, _SpooledObservationMapping)
    assert result.evidence_transport_bytes == len(result.evidence_json)
    consumed = result.without_evidence_transport()
    assert consumed.evidence_json == b""
    assert consumed.evidence_transport_bytes == result.evidence_transport_bytes
    try:
        validated = validate_recurrence_numerical_current_application(
            consumed,
            plan,
            baseline_plan=plan,
            precision_digits=80,
            seed=53,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )
        persisted = validated.to_json_dict()["persisted_numerical_evidence"]
        assert persisted["generation_evidence_encoding"] == ("zlib-canonical-json-v1")
        assert validated.application_validation["status"] == "verified"
    finally:
        result.close()
        result.close()
    with pytest.raises(RuntimeError, match="spool is closed"):
        candidate_spool[0]
    with pytest.raises(RuntimeError, match="spool is closed"):
        verification_spool[0]


def test_spooled_discovery_preserves_global_equal_opposite_and_zero_relations() -> None:
    plan = _signed_relation_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        _points(101),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=67,
        domain="independent-verification-current-probes-v1",
    )
    expected, _expected_report = _discover_relations(
        plan.sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=80,
        seed=67,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
        color_accuracy="lc",
    )
    candidate_spool = _SpooledObservationMapping(
        candidate.observations,
        candidate_indexes=_build_candidate_indexes(
            plan.sections,
            candidate.observations,
        ),
    )
    verification_spool = _SpooledObservationMapping(
        verification.observations,
    )
    try:
        actual, _actual_report = _discover_relations(
            plan.sections,
            replace(candidate, observations=candidate_spool),
            replace(verification, observations=verification_spool),
            source_semantics_sha256=source_digest,
            precision_digits=80,
            seed=67,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
            color_accuracy="lc",
        )
    finally:
        candidate_spool.close()
        verification_spool.close()

    assert actual == expected
    assert {
        (certificate.current_id, certificate.relation_kind) for certificate in actual
    } >= {(3, "equal"), (4, "opposite"), (5, "zero")}


def test_discovery_reads_each_candidate_current_once_and_reuses_residual_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _no_relation_plan()
    source_digest = recurrence_numerical_source_semantics_sha256(plan.sections)
    candidate = capture_recurrence_current_observations(
        plan,
        _points(1),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=71,
        domain="candidate-current-probes-v1",
    )
    verification = capture_recurrence_current_observations(
        plan,
        _points(101),
        precision_digits=80,
        source_semantics_sha256=source_digest,
        seed=71,
        domain="independent-verification-current-probes-v1",
    )
    candidate_indexes = _build_candidate_indexes(
        plan.sections,
        candidate.observations,
    )
    expected_certificates, expected_report = _discover_relations(
        plan.sections,
        candidate,
        verification,
        source_semantics_sha256=source_digest,
        precision_digits=80,
        seed=71,
        relative_tolerance=1.0e-60,
        absolute_tolerance=1.0e-70,
        color_accuracy="lc",
        candidate_indexes=candidate_indexes,
    )
    spool = _SpooledObservationMapping(candidate.observations)

    class CountingRows(
        Mapping[int, tuple[tuple[Decimal, Decimal], ...]],
    ):
        def __init__(self) -> None:
            self.reads: dict[int, int] = {}

        def __getitem__(
            self,
            current_id: int,
        ) -> tuple[tuple[Decimal, Decimal], ...]:
            self.reads[current_id] = self.reads.get(current_id, 0) + 1
            return spool[current_id]

        def __iter__(self) -> Iterator[int]:
            return iter(spool)

        def __len__(self) -> int:
            return len(spool)

    rows = CountingRows()
    fraction_string_calls = 0
    fraction_string = recurrence_warmup._fraction_string

    def counted_fraction_string(value: Fraction) -> str:
        nonlocal fraction_string_calls
        fraction_string_calls += 1
        return fraction_string(value)

    monkeypatch.setattr(
        recurrence_warmup,
        "_fraction_string",
        counted_fraction_string,
    )
    try:
        actual_certificates, actual_report = _discover_relations(
            plan.sections,
            replace(candidate, observations=rows),
            verification,
            source_semantics_sha256=source_digest,
            precision_digits=80,
            seed=71,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
            color_accuracy="lc",
            candidate_indexes=candidate_indexes,
        )
    finally:
        spool.close()

    assert actual_certificates == expected_certificates == ()
    assert actual_report == expected_report
    assert rows.reads == {2: 1, 3: 1}
    assert fraction_string_calls == 3 * (
        int(actual_report["tested_hypothesis_count"]) + 1
    )


def test_dynamic_raw_geometry_rejects_adversarial_sizes_and_reserves_wire() -> None:
    with pytest.raises(ValueError, match="memory envelope"):
        _raw_evidence_wire_byte_limit(
            scalar_count=1 << 63,
            row_count=1 << 63,
        )

    maximum = _raw_evidence_wire_byte_limit(
        scalar_count=0,
        row_count=0,
    )
    assert maximum == _MAX_RAW_EVIDENCE_BYTES
    assert maximum >= _MIN_RAW_EVIDENCE_WIRE_BYTES


@pytest.mark.parametrize("text", ("1e+1000000000", "1e-1000000000"))
def test_raw_evidence_rejects_huge_decimal_exponents_before_formatting(
    text: str,
) -> None:
    class DecimalThatMustNotFormat(Decimal):
        def __format__(self, _format_spec: str, /) -> str:
            raise AssertionError("unbounded fixed-point formatting was reached")

    with pytest.raises(ValueError, match="raw evidence scalar boundary"):
        _decimal_string(DecimalThatMustNotFormat(text))


def test_shared_candidate_index_is_complete_at_exact_signed_boundaries() -> None:
    large = Fraction(10**100)
    observations = {
        0: ((Fraction(100, 9), Fraction(0)), (large, Fraction(0))),
        1: ((Fraction(-100, 9), Fraction(0)), (-large, Fraction(0))),
        2: ((Fraction(1_000), Fraction(0)), (Fraction(0), Fraction(0))),
        4: ((Fraction(10), Fraction(0)), (large, Fraction(0))),
    }
    index = build_numerical_observation_candidate_index(
        (0, 1, 2, 4),
        observations,
        normalize=Fraction,
    )
    assert (index.observation_index, index.scalar_component) == (0, 0)
    common = {
        "current_id": 4,
        "relative_tolerance": Fraction(1, 10),
        "absolute_tolerance": Fraction(0),
        "normalize": Fraction,
    }
    assert numerical_observation_tolerance_window_ids(
        index,
        observations[4],
        relation_kind="equal",
        **common,
    ) == (0,)
    assert numerical_observation_tolerance_window_ids(
        index,
        observations[4],
        relation_kind="opposite",
        **common,
    ) == (1,)
    # The unrelated 10**100 component has zero residual for the accepted
    # signed pairs and cannot enlarge the selected pair's tolerance window.
    assert abs(Fraction(10) - Fraction(100, 9)) == Fraction(10, 9)
    assert Fraction(1, 10) * Fraction(100, 9) == Fraction(10, 9)


def test_rejected_diagnostic_full_digest_binds_unretained_tail() -> None:
    retained = tuple({"current_id": index} for index in range(16))
    root = "00" * 32
    common_tail = _advance_rejection_digest(root, {"current_id": 16})
    left_digest = _advance_rejection_digest(common_tail, {"current_id": 17})
    right_digest = _advance_rejection_digest(common_tail, {"current_id": 18})
    left = _rejected_candidate_diagnostics(
        retained,
        total_rejected=18,
        full_census={"tested": 18},
        full_rejection_sha256=left_digest,
    )
    right = _rejected_candidate_diagnostics(
        retained,
        total_rejected=18,
        full_census={"tested": 18},
        full_rejection_sha256=right_digest,
    )
    assert left["retained_sha256"] == right["retained_sha256"]
    assert left["full_census_sha256"] == right["full_census_sha256"]
    assert left["full_rejection_sha256"] != right["full_rejection_sha256"]
    assert left["truncated"] is True


@pytest.mark.parametrize(
    ("relative_tolerance", "absolute_tolerance"),
    (
        (-0.0, 1.0e-70),
        (1.0e-60, -0.0),
        (float("nan"), 1.0e-70),
        (1.0e-60, float("inf")),
    ),
)
def test_recurrence_raw_evidence_rejects_noncanonical_tolerances_before_capture(
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="tolerances are invalid"):
        run_recurrence_numerical_current_warmup(
            _topology_replay_plan(),
            candidate_points=_points(1),
            verification_points=_points(101),
            mode="certified-reuse",
            color_accuracy="lc",
            precision_digits=80,
            seed=53,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )


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
    lane = result.to_json_dict()
    discovery = lane["discovery"]
    application = lane["application"]
    assert isinstance(discovery, dict)
    assert isinstance(application, dict)
    native = {
        "requested_mode": "certified-reuse",
        "state": "native-applied",
        "exact_certified_relation_count": 1,
        "applied_relation_count": 1,
        "certificate_count": 1,
        "numerical_candidate_count": discovery["numerical_candidate_count"],
        "uncertified_candidate_count": discovery["verification_rejected_count"],
        "rejected_hypothesis_count": discovery["rejected_hypothesis_count"],
        "certificate_replay": application["certificate_replay"],
        "probe": {
            "probe_count": lane["candidate_capture"]["point_count"],
            "verification_probe_count": (lane["verification_capture"]["point_count"]),
            "verification_rejected_count": discovery["verification_rejected_count"],
            "tested_hypothesis_count": discovery["tested_hypothesis_count"],
            "runtime_parameter_schema_sha256": lane["candidate_capture"][
                "runtime_parameter_schema_sha256"
            ],
            "candidate_observation_batch_sha256": lane["candidate_capture"][
                "observation_batch_sha256"
            ],
            "verification_observation_batch_sha256": lane["verification_capture"][
                "observation_batch_sha256"
            ],
            "decision_sha256": discovery["decision_sha256"],
            "rejection_decision_sha256": discovery["rejection_decision_sha256"],
        },
        "rejected_candidate_diagnostics": {
            "total_rejected_hypothesis_count": discovery["rejected_hypothesis_count"],
            "full_rejection_sha256": discovery["rejection_decision_sha256"],
        },
    }
    runtime_inspection, aggregate = _recurrence_relation_reporting(
        {"relation_discovery": native},
        mode="certified-reuse",
        lane_report=lane,
    )
    assert "relation_discovery" not in runtime_inspection
    assert aggregate["execution_mode"] == "recurrence"
    assert aggregate["lanes"]["primary"]["applied_relation_count"] == 1
    assert aggregate["certified_relation_count"] == 1
    assert aggregate["applied_relation_count"] == 1
    assert aggregate["warning"]["required"] is True
    assert aggregate["warning"]["emit"] == "once-per-generated-artifact"
    assert aggregate["native_relation_application"] == native

    for mutation in (
        {"requested_mode": "diagnostic"},
        {"exact_certified_relation_count": 0},
        {"applied_relation_count": 0},
        {"certificate_count": 0},
        {
            "certificate_replay": {
                **native["certificate_replay"],
                "certificate_set_sha256": "0" * 64,
            }
        },
        {
            "probe": {
                **native["probe"],
                "verification_probe_count": 3,
            }
        },
    ):
        drifted = {**native, **mutation}
        with pytest.raises(GenerationError):
            _recurrence_relation_reporting(
                {"relation_discovery": drifted},
                mode="certified-reuse",
                lane_report=lane,
            )

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
            baseline_plan=_topology_replay_plan(),
            precision_digits=80,
            seed=79,
            relative_tolerance=1.0e-60,
            absolute_tolerance=1.0e-70,
        )
    except (ArtifactError, ValueError):
        pass
    else:  # pragma: no cover - fail-closed contract
        raise AssertionError("current-dimension drift was accepted")


def test_recurrence_service_forwards_complete_numerical_contract_to_pyo3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def binding(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        Path(str(args[5])).write_bytes(b"pacbin")
        return object()

    native = SimpleNamespace(_lower_recurrence_direct_v2=binding)
    monkeypatch.setattr(
        generation_service.importlib,
        "import_module",
        lambda _name: native,
    )
    monkeypatch.setattr(
        generation_service,
        "verify_native_module",
        lambda _module: None,
    )
    monkeypatch.setattr(
        generation_service,
        "active_native_source_identity",
        lambda: ("source-revision", "a" * 64),
    )
    monkeypatch.setattr(
        generation_service,
        "_validate_rust_recurrence_lowering_result",
        lambda *_args, **_kwargs: {
            "inspection_summary": {},
            "resolved_helicities": (),
            "amplitude_destinations": (),
            "exact_sections": {},
            "generation_profile": {"serialized_bytes": {"container": len(b"pacbin")}},
            "member_count": 1,
            "unpacked_size_bytes": 6,
            "index_sha256": "b" * 64,
        },
    )
    builder = SimpleNamespace(digest="c" * 64, layout="topology-replay")
    template = SimpleNamespace(digest="d" * 64)
    catalog = SimpleNamespace(
        catalog_digest="e" * 64,
        to_dict=lambda: {"z": 1, "a": 2},
    )

    def progress(_payload: object) -> None:
        pass

    evidence = b'{"raw":"evidence"}'
    destination = tmp_path / "recurrence-runtime.pacbin"

    output = generation_service._invoke_rust_recurrence_lowering_v2(
        builder,
        template,
        catalog,
        "f" * 64,
        "1" * 64,
        destination,
        point_tile_size=17,
        workspace_mib=23,
        relation_discovery_mode="certified-reuse",
        relation_discovery_precision_digits=101,
        relation_discovery_probe_count=3,
        relation_discovery_verification_probe_count=5,
        relation_discovery_relative_tolerance=1.25e-70,
        relation_discovery_absolute_tolerance=2.5e-80,
        relation_discovery_seed=123456789,
        color_accuracy="nlc",
        relation_discovery_evidence_json=evidence,
        progress_callback=progress,
    )
    cached_destination = tmp_path / "recurrence-runtime-cached.pacbin"
    cached_catalog = SimpleNamespace(
        catalog_digest="e" * 64,
        to_dict=lambda: pytest.fail("cached direct-template JSON was rebuilt"),
    )
    cached_output = generation_service._invoke_rust_recurrence_lowering_v2(
        builder,
        template,
        cached_catalog,
        "f" * 64,
        "1" * 64,
        cached_destination,
        point_tile_size=17,
        workspace_mib=23,
        relation_discovery_mode="certified-reuse",
        relation_discovery_precision_digits=101,
        relation_discovery_probe_count=3,
        relation_discovery_verification_probe_count=5,
        relation_discovery_relative_tolerance=1.25e-70,
        relation_discovery_absolute_tolerance=2.5e-80,
        relation_discovery_seed=123456789,
        color_accuracy="nlc",
        relation_discovery_evidence_json=evidence,
        progress_callback=progress,
        direct_template_catalog_json=b'{"a":2,"z":1}',
    )

    assert output.payload_path == destination
    assert output.generation_profile == {
        "serialized_bytes": {"container": len(b"pacbin")}
    }
    assert cached_output.payload_path == cached_destination
    assert len(calls) == 2
    args, kwargs = calls[0]
    assert args[:2] == (builder, template)
    assert args[2] == b'{"a":2,"z":1}'
    assert args[3:5] == ("f" * 64, "1" * 64)
    assert args[5] == str(destination)
    assert kwargs == {
        "source_revision": "source-revision",
        "native_build_inputs_sha256": "a" * 64,
        "point_tile_size": 17,
        "workspace_mib": 23,
        "relation_discovery_mode": "certified-reuse",
        "relation_discovery_precision_digits": 101,
        "relation_discovery_probe_count": 3,
        "relation_discovery_verification_probe_count": 5,
        "relation_discovery_relative_tolerance": 1.25e-70,
        "relation_discovery_absolute_tolerance": 2.5e-80,
        "relation_discovery_seed": 123456789,
        "color_accuracy": "nlc",
        "relation_discovery_evidence_json": evidence,
        "progress_callback": progress,
    }
    cached_args, cached_kwargs = calls[1]
    assert cached_args[2] == args[2]
    assert cached_kwargs == kwargs
