# SPDX-License-Identifier: 0BSD
"""High-precision current snapshots for bounded relation-discovery warm-up."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from math import cos, isfinite, pi, sin, sqrt
from typing import Any

from ..evaluators.symbolica_compile import _compile_symbolica_outputs
from ..evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from ..models.base import Model
from ..runtime.symbolica_exact import (
    _decimal,
    _fill_momenta,
    _fill_sources,
)
from .contracts import StageCompilationInput
from .dag_types import GenericDAG
from .runtime_schema import build_runtime_expression_schema
from .stage_planning import build_generic_stage_compiler_blueprint
from .stage_types import GenericCompiledStageBlueprint
from .validation import ValidationPointRecord, build_validation_point

_ComplexDecimal = tuple[Decimal, Decimal]
_CAPTURE_ABI = "pyamplicol-generic-dag-current-observation-capture-v1"


@dataclass(frozen=True, slots=True)
class GenericDAGCurrentObservationCapture:
    """Complete point-major current observations and replay provenance."""

    precision_digits: int
    points: tuple[ValidationPointRecord, ...]
    point_sha256s: tuple[str, ...]
    kinematic_sha256s: tuple[str, ...]
    observations: Mapping[int, tuple[_ComplexDecimal, ...]]
    runtime_schema_sha256: str
    source_dag_sha256: str
    observation_batch_sha256: str

    @property
    def point_count(self) -> int:
        return len(self.point_sha256s)

    @property
    def current_count(self) -> int:
        return len(self.observations)

    def to_provenance_dict(self) -> dict[str, object]:
        return {
            "abi": _CAPTURE_ABI,
            "precision_digits": self.precision_digits,
            "point_count": self.point_count,
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "points": [point.to_mapping() for point in self.points],
            "current_count": self.current_count,
            "runtime_schema_sha256": self.runtime_schema_sha256,
            "source_dag_sha256": self.source_dag_sha256,
            "observation_batch_sha256": self.observation_batch_sha256,
            "complete_current_components": True,
            "point_major": True,
            "evaluator": "symbolica-interpreted-high-precision-stage-replay",
        }


def build_numerical_current_probe_points(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str,
    seed: int,
    candidate_count: int,
    verification_count: int,
) -> tuple[
    tuple[ValidationPointRecord, ...],
    tuple[ValidationPointRecord, ...],
]:
    """Build deterministic domain-separated physical warm-up points."""

    if (
        type(seed) is not int
        or seed < 0
        or type(candidate_count) is not int
        or candidate_count < 2
        or type(verification_count) is not int
        or verification_count < 2
    ):
        raise ValueError("numerical current probe-point contract is invalid")

    def domain_points(
        domain: str,
        count: int,
    ) -> tuple[ValidationPointRecord, ...]:
        return tuple(
            _rotate_validation_point(
                build_validation_point(
                    dag,
                    model,
                    process_id=process_id,
                    seed=_domain_seed(seed, domain=domain, index=index),
                ),
                rotation_seed=_domain_seed(
                    seed,
                    domain=f"{domain}:spatial-rotation-v1",
                    index=index,
                ),
            )
            for index in range(count)
        )

    candidate = domain_points("candidate-current-probes-v1", candidate_count)
    verification = domain_points(
        "independent-verification-current-probes-v1",
        verification_count,
    )
    if any(not point.available for point in (*candidate, *verification)):
        errors = tuple(
            point.error
            for point in (*candidate, *verification)
            if not point.available
        )
        raise ValueError(
            "numerical current probe-point generation failed: "
            + "; ".join(str(error) for error in errors)
        )
    candidate_hashes = {_kinematic_sha256(point) for point in candidate}
    verification_hashes = {
        _kinematic_sha256(point) for point in verification
    }
    if (
        len(candidate_hashes) != len(candidate)
        or len(verification_hashes) != len(verification)
        or not candidate_hashes.isdisjoint(verification_hashes)
    ):
        raise ValueError(
            "numerical current probe domains do not contain distinct, "
            "independent physical momentum records"
        )
    return candidate, verification


def capture_generic_dag_current_observations(
    dag: GenericDAG,
    model: Model,
    points: Sequence[ValidationPointRecord],
    *,
    precision_digits: int,
) -> GenericDAGCurrentObservationCapture:
    """Evaluate and snapshot every current at every supplied physical point.

    The evaluators remain interpreted Symbolica states: this warm-up does not
    launch a JIT, C++, ASM, or native compiler.  Current components are copied
    immediately after their producing stage, before value-slot recycling can
    overwrite them.
    """

    if type(precision_digits) is not int or precision_digits < 80:
        raise ValueError(
            "numerical current capture precision must be at least 80 digits"
        )
    point_records = tuple(points)
    if len(point_records) < 2:
        raise ValueError(
            "numerical current capture requires at least two physical points"
        )
    if any(
        not isinstance(point, ValidationPointRecord) or not point.available
        for point in point_records
    ):
        raise ValueError(
            "numerical current capture received an unavailable point"
        )
    if any(point.process != dag.process.process for point in point_records):
        raise ValueError(
            "numerical current capture point process does not match the DAG"
        )

    runtime_schema = build_runtime_expression_schema(
        dag,
        model,
        process_id=point_records[0].process_id,
    )
    schema = runtime_schema.to_mapping()
    blueprint = build_generic_stage_compiler_blueprint(
        StageCompilationInput(dag, model, runtime_schema)
    )
    if not blueprint.expression_ready:
        raise ValueError(
            "numerical current capture cannot lower stage expressions: "
            + "; ".join(blueprint.blockers)
        )
    settings = SymbolicaEvaluatorSettings(
        iterations=1,
        cpe_iterations=0,
        n_cores=1,
        direct_translation=False,
        jit_direct_translation=False,
        jit_optimization_level=0,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        compiled_chunk_compile_workers=1,
    )
    stage_evaluators = tuple(
        _build_interpreted_stage_evaluator(stage, settings=settings)
        for stage in blueprint.stages
    )
    model_parameters = _runtime_model_parameter_values(schema)
    point_hashes = tuple(
        _validation_point_sha256(point) for point in point_records
    )
    kinematic_hashes = tuple(
        _kinematic_sha256(point) for point in point_records
    )
    if len(set(kinematic_hashes)) != len(kinematic_hashes):
        raise ValueError(
            "numerical current capture requires distinct physical momenta"
        )
    captured_by_current: dict[int, list[_ComplexDecimal]] = {
        current.id: [] for current in dag.currents
    }
    working_precision = precision_digits + 16
    with localcontext() as context:
        context.prec = working_precision
        context.rounding = ROUND_HALF_EVEN
        for point in point_records:
            point_values = tuple(
                tuple(Decimal.from_float(float(value)) for value in momentum)
                for momentum in point.four_vectors
            )
            point_capture = _capture_one_point(
                dag,
                schema,
                blueprint.stages,
                stage_evaluators,
                point_values,
                model_parameters,
                precision=working_precision,
            )
            for current in dag.currents:
                values = point_capture[current.id]
                if len(values) != current.dimension:
                    raise ValueError(
                        f"numerical current {current.id} capture has "
                        f"{len(values)} components, expected {current.dimension}"
                    )
                captured_by_current[current.id].extend(values)

    observations = {
        current_id: tuple(values)
        for current_id, values in captured_by_current.items()
    }
    if any(
        not real.is_finite() or not imaginary.is_finite()
        for values in observations.values()
        for real, imaginary in values
    ):
        raise ValueError(
            "numerical current capture produced a non-finite component"
        )
    source_dag_digest = _canonical_sha256(dag.to_json_dict())
    observation_digest = _canonical_sha256(
        {
            "point_sha256s": list(point_hashes),
            "currents": [
                {
                    "current_id": current_id,
                    "values": [
                        [_decimal_string(real), _decimal_string(imaginary)]
                        for real, imaginary in observations[current_id]
                    ],
                }
                for current_id in sorted(observations)
            ],
        }
    )
    return GenericDAGCurrentObservationCapture(
        precision_digits=precision_digits,
        points=point_records,
        point_sha256s=point_hashes,
        kinematic_sha256s=kinematic_hashes,
        observations=observations,
        runtime_schema_sha256=runtime_schema.sha256,
        source_dag_sha256=source_dag_digest,
        observation_batch_sha256=observation_digest,
    )


def validate_independent_current_observation_captures(
    candidate: GenericDAGCurrentObservationCapture,
    verification: GenericDAGCurrentObservationCapture,
) -> None:
    """Fail closed unless two capture batches form one independent contract."""

    if (
        candidate.precision_digits != verification.precision_digits
        or candidate.runtime_schema_sha256
        != verification.runtime_schema_sha256
        or candidate.source_dag_sha256 != verification.source_dag_sha256
        or set(candidate.observations) != set(verification.observations)
        or candidate.point_count < 2
        or verification.point_count < 2
        or len(set(candidate.point_sha256s)) != candidate.point_count
        or len(set(verification.point_sha256s)) != verification.point_count
        or not set(candidate.point_sha256s).isdisjoint(
            verification.point_sha256s
        )
        or len(set(candidate.kinematic_sha256s)) != candidate.point_count
        or len(set(verification.kinematic_sha256s))
        != verification.point_count
        or not set(candidate.kinematic_sha256s).isdisjoint(
            verification.kinematic_sha256s
        )
    ):
        raise ValueError(
            "numerical current candidate and verification captures do not "
            "form independent replay domains"
        )
    for current_id in candidate.observations:
        candidate_values = candidate.observations[current_id]
        verification_values = verification.observations[current_id]
        if (
            len(candidate_values) % candidate.point_count != 0
            or len(verification_values) % verification.point_count != 0
            or len(candidate_values) // candidate.point_count
            != len(verification_values) // verification.point_count
        ):
            raise ValueError(
                f"numerical current {current_id} capture widths drifted "
                "between candidate and verification domains"
            )


def _build_interpreted_stage_evaluator(
    stage: GenericCompiledStageBlueprint,
    *,
    settings: SymbolicaEvaluatorSettings,
) -> Any:
    return _compile_symbolica_outputs(
        stage.output_expressions,
        list(stage.parameter_symbols),
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        functions={
            (function, arguments): body
            for function, arguments, body in stage.symbolica_functions
        },
        real_params=stage.real_valued_inputs,
        symbolica_settings=settings,
        jit_compile=False,
        label=f"numerical_current_warmup_stage_{stage.stage_index}",
        output_partitions=(),
    )


def _capture_one_point(
    dag: GenericDAG,
    schema: Mapping[str, object],
    stages: tuple[GenericCompiledStageBlueprint, ...],
    evaluators: tuple[Any, ...],
    point: tuple[tuple[Decimal, ...], ...],
    model_parameters: tuple[Decimal, ...],
    *,
    precision: int,
) -> dict[int, tuple[_ComplexDecimal, ...]]:
    layout = _mapping(schema.get("parameter_layout"), "parameter layout")
    value_count = _integer(layout.get("value_component_count"))
    momentum_count = _integer(layout.get("momentum_parameter_count"))
    flattened_count = _integer(layout.get("parameter_count_if_flattened"))
    model_start = value_count + momentum_count
    state = [
        (Decimal(0), Decimal(0))
        for _index in range(
            max(flattened_count, model_start + len(model_parameters))
        )
    ]
    _fill_sources(state, point, schema, model_parameters)
    _fill_momenta(state, point, schema)
    for index, value in enumerate(model_parameters):
        state[model_start + index] = (value, Decimal(0))

    captured: dict[int, tuple[_ComplexDecimal, ...]] = {}
    source_fill = _mapping(schema.get("source_fill"), "source fill")
    sources = _sequence(source_fill.get("sources"), "source rows")
    for raw_source in sources:
        source = _mapping(raw_source, "source row")
        slot = _mapping(source.get("value_slot"), "source value slot")
        current_id = _integer(source.get("current_id"))
        start = _integer(slot.get("component_start"))
        stop = _integer(slot.get("component_stop"))
        if current_id in captured or not 0 <= start <= stop <= len(state):
            raise ValueError(
                "numerical current source capture has an invalid slot"
            )
        captured[current_id] = tuple(state[start:stop])

    for stage, evaluator in zip(stages, evaluators, strict=True):
        inputs: list[_ComplexDecimal | None] = [None] * stage.parameter_count
        for component in stage.input_components:
            if (
                not 0 <= component.parameter_index < len(inputs)
                or not 0 <= component.global_component < len(state)
                or inputs[component.parameter_index] is not None
            ):
                raise ValueError(
                    "numerical current stage input mapping is invalid"
                )
            inputs[component.parameter_index] = state[
                component.global_component
            ]
        if any(value is None for value in inputs):
            raise ValueError(
                "numerical current stage input mapping is incomplete"
            )
        outputs = _evaluate_interpreted_stage(
            evaluator,
            tuple(value for value in inputs if value is not None),
            precision=precision,
        )
        if len(outputs) != stage.output_length:
            raise ValueError(
                "numerical current stage evaluator returned the wrong width"
            )
        for slot in stage.output_slots:
            values = outputs[slot.output_start : slot.output_stop]
            if (
                slot.current_id in captured
                or len(values) != slot.component_stop - slot.component_start
                or not 0
                <= slot.component_start
                <= slot.component_stop
                <= len(state)
            ):
                raise ValueError(
                    "numerical current stage output mapping is invalid"
                )
            state[slot.component_start : slot.component_stop] = values
            captured[slot.current_id] = values
    if set(captured) != {current.id for current in dag.currents}:
        raise ValueError(
            "numerical current capture did not visit every current exactly once"
        )
    return captured


def _evaluate_interpreted_stage(
    evaluator: Any,
    values: tuple[_ComplexDecimal, ...],
    *,
    precision: int,
) -> tuple[_ComplexDecimal, ...]:
    source = getattr(evaluator, "_source_evaluator", None)
    evaluate = getattr(source, "evaluate_complex_with_prec", None)
    if not callable(evaluate):
        raise ValueError(
            "numerical current warm-up evaluator lacks high-precision replay"
        )
    try:
        raw = evaluate(values, precision)
    except Exception as error:
        raise ValueError(
            f"numerical current high-precision stage replay failed: {error}"
        ) from error
    return tuple(
        (
            _decimal(value[0], "numerical current real component"),
            _decimal(value[1], "numerical current imaginary component"),
        )
        for value in raw
    )


def _runtime_model_parameter_values(
    schema: Mapping[str, object],
) -> tuple[Decimal, ...]:
    records = _sequence(schema.get("model_parameters", ()), "model parameters")
    ordered: list[Decimal | None] = [None] * len(records)
    for raw_record in records:
        record = _mapping(raw_record, "model parameter")
        index = _integer(record.get("parameter_index"))
        if (
            not 0 <= index < len(ordered)
            or ordered[index] is not None
            or "default" not in record
        ):
            raise ValueError(
                "numerical current model-parameter layout is invalid"
            )
        value = float(record["default"])
        if not isfinite(value):
            raise ValueError(
                "numerical current model parameter is non-finite"
            )
        ordered[index] = Decimal.from_float(value)
    if any(value is None for value in ordered):
        raise ValueError(
            "numerical current model-parameter layout is incomplete"
        )
    return tuple(value for value in ordered if value is not None)


def _validation_point_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(point.to_mapping())


def _kinematic_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(
        [
            {
                "pdg": pdg,
                "momentum": [format(float(value), ".17g") for value in momentum],
            }
            for pdg, momentum in point.particles
        ]
    )


def _rotate_validation_point(
    point: ValidationPointRecord,
    *,
    rotation_seed: int,
) -> ValidationPointRecord:
    """Apply a deterministic proper rotation to one physical event."""

    if not point.available:
        return point
    digest = hashlib.sha256(
        f"{rotation_seed}:proper-spatial-rotation-v1".encode("ascii")
    ).digest()
    denominator = float(1 << 64)
    azimuth = 2.0 * pi * (
        (int.from_bytes(digest[:8], "big") + 0.5) / denominator
    )
    cosine_polar = (
        2.0 * ((int.from_bytes(digest[8:16], "big") + 0.5) / denominator)
        - 1.0
    )
    sine_polar = sqrt(max(0.0, 1.0 - cosine_polar * cosine_polar))
    cos_azimuth = cos(azimuth)
    sin_azimuth = sin(azimuth)
    # Rz(azimuth) Ry(polar), a proper orthogonal map taking +z to the
    # deterministic direction above.
    rotation = (
        (
            cos_azimuth * cosine_polar,
            -sin_azimuth,
            cos_azimuth * sine_polar,
        ),
        (
            sin_azimuth * cosine_polar,
            cos_azimuth,
            sin_azimuth * sine_polar,
        ),
        (-sine_polar, 0.0, cosine_polar),
    )

    def rotated(
        momentum: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        energy, x_component, y_component, z_component = momentum
        spatial = (x_component, y_component, z_component)
        return (
            energy,
            *tuple(
                sum(row[column] * spatial[column] for column in range(3))
                for row in rotation
            ),
        )

    return replace(
        point,
        particles=tuple(
            (pdg, rotated(momentum)) for pdg, momentum in point.particles
        ),
    )


def _domain_seed(seed: int, *, domain: str, index: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{domain}:{index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _decimal_string(value: Decimal) -> str:
    return "0" if value == 0 else str(value.normalize())


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"numerical current {label} is not an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"numerical current {label} is not a sequence")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("numerical current layout integer is invalid")
    return value


__all__ = [
    "GenericDAGCurrentObservationCapture",
    "build_numerical_current_probe_points",
    "capture_generic_dag_current_observations",
    "validate_independent_current_observation_captures",
]
