# SPDX-License-Identifier: 0BSD
"""Real high-precision warm-up observations for numerical current reuse."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from math import isclose
from typing import Any

import pytest

from pyamplicol.generation import numerical_current_warmup
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_equivalence import (
    apply_numerical_current_relation_certificates,
    discover_generic_dag_numerical_current_relations,
)
from pyamplicol.generation.dag_types import GenericDAG
from pyamplicol.generation.numerical_current_warmup import (
    build_numerical_current_probe_points,
    capture_generic_dag_current_observations,
    validate_independent_current_observation_captures,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir

_PRECISION_DIGITS = 96
_SEED = 17


@pytest.fixture(scope="module")
def z_dag_and_model() -> tuple[GenericDAG, BuiltinSMModel]:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z", color_accuracy="lc"),
        model=model,
    )
    return dag, model


def _points(
    dag: GenericDAG,
    model: BuiltinSMModel,
) -> tuple[
    tuple[ValidationPointRecord, ...],
    tuple[ValidationPointRecord, ...],
]:
    return build_numerical_current_probe_points(
        dag,
        model,
        process_id="ddbar_z_numerical_warmup",
        seed=_SEED,
        candidate_count=4,
        verification_count=4,
    )


def test_probe_points_are_deterministic_physically_distinct_domains(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
) -> None:
    dag, model = z_dag_and_model
    candidate, verification = _points(dag, model)

    assert (candidate, verification) == _points(dag, model)
    all_points = (*candidate, *verification)
    assert len({point.seed for point in all_points}) == 8
    assert len({point.four_vectors for point in all_points}) == 8
    for point in all_points:
        incoming = point.four_vectors[:2]
        outgoing = point.four_vectors[2:]
        for component in range(4):
            assert isclose(
                sum(momentum[component] for momentum in incoming),
                sum(momentum[component] for momentum in outgoing),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )


def test_capture_uses_only_interpreted_high_precision_stage_replay(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag, model = z_dag_and_model
    candidate, _verification = _points(dag, model)
    real_compile = numerical_current_warmup._compile_symbolica_outputs
    compile_contracts: list[dict[str, Any]] = []

    def guarded_compile(*args: object, **kwargs: Any) -> Any:
        assert kwargs["jit_compile"] is False
        settings = kwargs["symbolica_settings"]
        assert settings.direct_translation is False
        assert settings.jit_direct_translation is False
        assert settings.jit_optimization_level == 0
        compile_contracts.append(dict(kwargs))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(
        numerical_current_warmup,
        "_compile_symbolica_outputs",
        guarded_compile,
    )
    capture = capture_generic_dag_current_observations(
        dag,
        model,
        candidate,
        precision_digits=_PRECISION_DIGITS,
    )

    assert compile_contracts
    assert capture.precision_digits == _PRECISION_DIGITS
    assert capture.point_count == 4
    assert capture.current_count == len(dag.currents)
    assert len(set(capture.kinematic_sha256s)) == 4
    assert set(capture.observations) == {
        current.id for current in dag.currents
    }
    for current in dag.currents:
        values = capture.observations[current.id]
        assert len(values) == capture.point_count * current.dimension
        assert all(
            isinstance(component, Decimal) and component.is_finite()
            for value in values
            for component in value
        )
    provenance = capture.to_provenance_dict()
    assert provenance["complete_current_components"] is True
    assert provenance["point_major"] is True
    assert provenance["points"] == [
        point.to_mapping() for point in candidate
    ]


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_real_capture_drives_authenticated_discovery_and_application(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
    execution_mode: str,
) -> None:
    dag, model = z_dag_and_model
    candidate_points, verification_points = _points(dag, model)
    candidate = capture_generic_dag_current_observations(
        dag,
        model,
        candidate_points,
        precision_digits=_PRECISION_DIGITS,
    )
    verification = capture_generic_dag_current_observations(
        dag,
        model,
        verification_points,
        precision_digits=_PRECISION_DIGITS,
    )
    validate_independent_current_observation_captures(
        candidate,
        verification,
    )

    discovery = discover_generic_dag_numerical_current_relations(
        dag,
        model,
        candidate_observations=candidate.observations,
        verification_observations=verification.observations,
        candidate_point_sha256s=candidate.point_sha256s,
        verification_point_sha256s=verification.point_sha256s,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        precision_digits=_PRECISION_DIGITS,
        seed=_SEED,
        relative_tolerance=1.0e-70,
        absolute_tolerance=1.0e-80,
    )
    assert discovery.report.inspected_current_count == len(dag.currents)
    assert (
        discovery.report.candidate_observation_batch_sha256
        == candidate.observation_batch_sha256
    )
    assert (
        discovery.report.verification_observation_batch_sha256
        == verification.observation_batch_sha256
    )

    application = apply_numerical_current_relation_certificates(
        dag,
        model,
        discovery.certificates,
        mode="certified-reuse",
        execution_mode=execution_mode,  # type: ignore[arg-type]
    )
    if discovery.certificates:
        assert application.report.state == "authenticated-numerical-applied"
        assert application.report.warning_required
        assert application.report.applied_relation_count == len(
            discovery.certificates
        )
    else:
        assert (
            discovery.report.state
            == application.report.state
            == "no_certified_numerical_relation"
        )
        assert not application.report.warning_required
        assert application.report.applied_relation_count == 0
        assert application.dag is dag


def test_capture_domains_fail_closed_on_replay_drift(
    z_dag_and_model: tuple[GenericDAG, BuiltinSMModel],
) -> None:
    dag, model = z_dag_and_model
    candidate_points, verification_points = _points(dag, model)
    with pytest.raises(ValueError, match="distinct physical momenta"):
        capture_generic_dag_current_observations(
            dag,
            model,
            (candidate_points[0], candidate_points[0]),
            precision_digits=_PRECISION_DIGITS,
        )
    with pytest.raises(ValueError, match="does not match"):
        capture_generic_dag_current_observations(
            dag,
            model,
            (
                replace(candidate_points[0], process="g g > z"),
                candidate_points[1],
            ),
            precision_digits=_PRECISION_DIGITS,
        )

    candidate = capture_generic_dag_current_observations(
        dag,
        model,
        candidate_points,
        precision_digits=_PRECISION_DIGITS,
    )
    verification = capture_generic_dag_current_observations(
        dag,
        model,
        verification_points,
        precision_digits=_PRECISION_DIGITS,
    )
    repeated_kinematics = replace(
        verification,
        kinematic_sha256s=candidate.kinematic_sha256s,
    )
    with pytest.raises(ValueError, match="independent replay domains"):
        validate_independent_current_observation_captures(
            candidate,
            repeated_kinematics,
        )

    current_id = min(candidate.observations)
    incomplete = replace(
        verification,
        observations={
            **verification.observations,
            current_id: verification.observations[current_id][:-1],
        },
    )
    with pytest.raises(ValueError, match="widths drifted"):
        validate_independent_current_observation_captures(
            candidate,
            incomplete,
        )
