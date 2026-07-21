# SPDX-License-Identifier: 0BSD
"""Lazy Symbolica-backed execution for non-f64 precision requests.

The process artifact is a trusted executable input. Evaluator states are loaded
with Symbolica's own ``Evaluator.load`` implementation; this module does not
decode or reinterpret Symbolica's serialization format.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, TypedDict, cast

from pyamplicol.api.errors import (
    ArtifactError,
    CompatibilityError,
    DependencyError,
    EvaluationError,
)
from pyamplicol.api.protocols import Momenta
from pyamplicol.api.results import ResolvedEvaluation
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import normalize_relative_path
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver

_ComplexDecimal = tuple[Decimal, Decimal]
_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_MINIMUM_SYMBOLICA_ARBITRARY_PRECISION = 40
_ARITHMETIC_GUARD_DIGITS = 8


class _RuntimeState(TypedDict):
    model_parameter_values: Sequence[object]
    normalization_factor: object


@dataclass(frozen=True, slots=True)
class _LcReplayRoute:
    source_index: int
    target_index: int
    weight: Decimal


@dataclass(frozen=True, slots=True)
class _LcReplayEntry:
    input_mapping: tuple[tuple[int, int], ...]
    routes: tuple[_LcReplayRoute, ...]


@dataclass(frozen=True, slots=True)
class _LcReplayPlan:
    entries: tuple[_LcReplayEntry, ...]
    helicity_count: int
    color_count: int


@dataclass(frozen=True, slots=True)
class _LcMaterializedSector:
    color_index: int
    reduction_weight: Decimal


@dataclass(frozen=True, slots=True)
class _LcReplaySectorRoute:
    physical_sector_id: int
    materialized_sector_id: int
    reduction_weight: Decimal
    residual: bool


@dataclass(frozen=True, slots=True)
class _ExactRuntimeSourceState:
    helicity: int
    chirality: int
    spin_state: int
    factor: _ComplexDecimal


@dataclass(frozen=True, slots=True)
class _ExactHelicitySchedule:
    physical_helicity_index: int
    selector_domain_id: int
    structural_zero: bool
    source_states: tuple[_ExactRuntimeSourceState | None, ...]
    root_factors: tuple[_ComplexDecimal | None, ...]


@dataclass(frozen=True, slots=True)
class _ExactHelicityPlan:
    schedules: tuple[_ExactHelicitySchedule, ...]


def _json_integer(value: object) -> int:
    return int(cast(int | float | str, value))


def _decimal(value: object, context: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise EvaluationError(f"{context} is not a decimal scalar") from exc
    if not result.is_finite():
        raise EvaluationError(f"{context} must be finite")
    return result


def _complex_zero() -> _ComplexDecimal:
    return (_ZERO, _ZERO)


def _complex_mul(left: _ComplexDecimal, right: _ComplexDecimal) -> _ComplexDecimal:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _upcast_decimal(value: Decimal, precision: int) -> Decimal:
    """Encode ``value`` with at least the precision requested from Symbolica.

    Symbolica honors the precision carried by each input ``Decimal``.  Values
    such as ``Decimal("500")`` otherwise enter an arbitrary-precision evaluator
    with only three significant digits and can reduce the precision of an
    entire instruction chain.  The extra zeroes intentionally do not invent
    information; they implement pyAmpliCol's documented upcast of input
    kinematics and intermediate stage values.
    """

    with localcontext() as context:
        context.prec = max(precision, len(value.as_tuple().digits), 1) + 2
        context.rounding = ROUND_HALF_EVEN
        if value.is_zero():
            return Decimal((value.as_tuple().sign, (0,), -precision))
        return Decimal(format(value, f".{precision - 1}E"))


def _upcast_complex_inputs(
    values: Sequence[_ComplexDecimal], precision: int
) -> tuple[_ComplexDecimal, ...]:
    return tuple(
        (
            _upcast_decimal(real, precision),
            _upcast_decimal(imaginary, precision),
        )
        for real, imaginary in values
    )


def _working_precision(requested_precision: int) -> int:
    # Symbolica uses a binary64 shortcut at 32 decimal digits. Stay above that
    # threshold for every request routed through the arbitrary-precision path.
    return max(
        requested_precision + _ARITHMETIC_GUARD_DIGITS,
        _MINIMUM_SYMBOLICA_ARBITRARY_PRECISION,
    )


def _sqrt(value: Decimal, context: str) -> Decimal:
    if value < 0:
        raise EvaluationError(f"{context} encountered a negative square root")
    return value.sqrt()


def _fortran_sign(value: Decimal, sign_source: Decimal) -> Decimal:
    return abs(value) if sign_source >= 0 else -abs(value)


@dataclass(slots=True)
class _ExactEvaluator:
    input_len: int
    evaluator: Any | None = None
    chunks: tuple[_ExactEvaluator, ...] = ()
    chunk_input_indices: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def load(
        cls,
        manifest: Mapping[str, object],
        root: Path,
        *,
        state_loader: Callable[[str], bytes] | None = None,
    ) -> _ExactEvaluator:
        kind = str(manifest.get("kind", ""))
        if kind == "chunked-symbolica-evaluator":
            raw_chunks = manifest.get("chunks")
            if isinstance(raw_chunks, str | bytes) or not isinstance(
                raw_chunks, Sequence
            ):
                raise ArtifactError("chunked evaluator has no chunk list")
            evaluators: list[_ExactEvaluator] = []
            for raw_chunk in raw_chunks:
                if not isinstance(raw_chunk, Mapping):
                    raise ArtifactError("chunked evaluator entry is not an object")
                evaluators.append(
                    cls.load(raw_chunk, root, state_loader=state_loader)
                )
            if not evaluators:
                raise ArtifactError("chunked evaluator has no evaluators")
            raw_input_len = manifest.get("input_len")
            raw_input_indices = manifest.get("chunk_input_indices")
            if raw_input_len is None and raw_input_indices is None:
                child_lengths = {evaluator.input_len for evaluator in evaluators}
                if len(child_lengths) != 1:
                    raise ArtifactError(
                        "legacy chunked evaluator children have inconsistent inputs"
                    )
                input_len = child_lengths.pop()
                input_indices = tuple(
                    tuple(range(input_len)) for _evaluator in evaluators
                )
            elif (
                isinstance(raw_input_len, int)
                and not isinstance(raw_input_len, bool)
                and isinstance(raw_input_indices, Sequence)
                and not isinstance(raw_input_indices, str | bytes)
            ):
                input_len = raw_input_len
                input_indices = tuple(
                    _exact_chunk_input_indices(indices, input_len)
                    for indices in raw_input_indices
                )
            else:
                raise ArtifactError("chunked evaluator input metadata is incomplete")
            if len(input_indices) != len(evaluators):
                raise ArtifactError(
                    "chunked evaluator input maps do not match evaluator chunks"
                )
            for evaluator, indices in zip(evaluators, input_indices, strict=True):
                if len(indices) != evaluator.input_len:
                    raise ArtifactError(
                        "chunked evaluator input map has inconsistent length"
                    )
            return cls(
                input_len=input_len,
                chunks=tuple(evaluators),
                chunk_input_indices=input_indices,
            )

        state_path = manifest.get("evaluator_state_path")
        if not isinstance(state_path, str) or not state_path:
            raise CompatibilityError(
                "higher-precision evaluation requires retained Symbolica evaluator "
                "state; regenerate this process artifact with the JIT backend"
            )
        if state_loader is None:
            path = (root / state_path).resolve(strict=False)
            try:
                path.relative_to(root.resolve(strict=True))
            except ValueError as exc:
                raise ArtifactError(
                    "Symbolica evaluator state escapes the process root"
                ) from exc
        else:
            normalized_state_path = normalize_relative_path(state_path)
            path = root / normalized_state_path
        try:
            from symbolica import Evaluator
        except ImportError as exc:
            raise DependencyError(
                "precision above 16 requires the Symbolica Python package; "
                "f64 SymJIT evaluation remains Symbolica-independent"
            ) from exc
        if state_loader is None:
            try:
                state = path.read_bytes()
            except OSError as exc:
                raise CompatibilityError(
                    f"could not read retained Symbolica evaluator state {path}: "
                    f"{exc}"
                ) from exc
        else:
            state = state_loader(state_path)
        try:
            raw_leaf_input_len = manifest.get("input_len")
            if isinstance(raw_leaf_input_len, bool) or not isinstance(
                raw_leaf_input_len, int
            ):
                raise ArtifactError("evaluator state has no valid input length")
            return cls(
                input_len=raw_leaf_input_len,
                evaluator=Evaluator.load(state),
            )
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise CompatibilityError(
                f"Symbolica could not load retained evaluator state {path}: {exc}"
            ) from exc

    def evaluate(
        self,
        values: Sequence[_ComplexDecimal],
        precision: int,
    ) -> tuple[_ComplexDecimal, ...]:
        prepared_values = _upcast_complex_inputs(values, precision)
        if len(prepared_values) != self.input_len:
            raise EvaluationError(
                "Symbolica high-precision evaluator input width is inconsistent"
            )
        return self._evaluate_prepared(prepared_values, precision)

    def _evaluate_prepared(
        self,
        values: tuple[_ComplexDecimal, ...],
        precision: int,
    ) -> tuple[_ComplexDecimal, ...]:
        if self.evaluator is not None:
            try:
                result = self.evaluator.evaluate_complex_with_prec(values, precision)
            except Exception as exc:
                raise EvaluationError(
                    f"Symbolica high-precision evaluator failed: {exc}"
                ) from exc
            return tuple(
                (
                    _decimal(value[0], "evaluator real output"),
                    _decimal(value[1], "evaluator imaginary output"),
                )
                for value in result
            )

        outputs: list[_ComplexDecimal] = []
        for evaluator, indices in zip(
            self.chunks,
            self.chunk_input_indices,
            strict=True,
        ):
            outputs.extend(
                evaluator._evaluate_prepared(
                    tuple(values[index] for index in indices),
                    precision,
                )
            )
        return tuple(outputs)


def _exact_chunk_input_indices(value: object, input_len: int) -> tuple[int, ...]:
    if (
        input_len < 0
        or isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
    ):
        raise ArtifactError("chunked evaluator input map is invalid")
    indices = tuple(value)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= input_len
        for index in indices
    ) or any(left >= right for left, right in pairwise(indices)):
        raise ArtifactError("chunked evaluator input map is invalid")
    return cast(tuple[int, ...], indices)


class SymbolicaExactExecutor:
    """Replay one verified schema-v3 execution plan through Symbolica states."""

    def __init__(self, artifact: Path, process_id: str, native_runtime: Any) -> None:
        self._artifact = artifact
        self._native_runtime = native_runtime
        manifest = load_manifest(artifact)
        self._payloads = ExactEvaluatorPayloadResolver(manifest)
        process, permutation = _selected_process(manifest.processes, process_id)
        representative_id = str(process["id"])
        self._representative_id = representative_id
        records = tuple(
            record
            for record in manifest.payloads
            if record.role == "evaluator-manifest"
            and record.process_id == representative_id
        )
        if len(records) != 1:
            raise ArtifactError(
                f"process {representative_id!r} must declare one evaluator manifest"
            )
        self._process_root = artifact / "processes" / representative_id
        self._process_prefix = normalize_relative_path(
            f"processes/{representative_id}"
        )
        try:
            self._execution = json.loads(
                (artifact / records[0].path).read_text(encoding="utf-8")
            )
            self._physics = json.loads(native_runtime.physics_json())
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"could not load exact-runtime metadata: {exc}"
            ) from exc
        if not isinstance(self._execution, dict) or not isinstance(self._physics, dict):
            raise ArtifactError("exact-runtime metadata is not an object")
        self._permutation = permutation
        self._lc_replay = _lc_replay_plan(
            self._execution,
            self._physics,
            permutation,
        )
        self._helicity_plan = _exact_helicity_plan(
            self._execution,
            self._physics,
            permutation,
        )
        self._stage_evaluators: tuple[_ExactEvaluator, ...] | None = None
        self._amplitude_evaluator: _ExactEvaluator | None = None

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        precision: int,
    ) -> ResolvedEvaluation:
        if isinstance(precision, bool) or not isinstance(precision, int):
            raise EvaluationError(
                "precision must be a positive integer number of decimal digits"
            )
        if precision < 1:
            raise EvaluationError(
                "precision must be a positive integer number of decimal digits"
            )
        working_precision = _working_precision(precision)
        points = _prepare_points(momenta, self._physics, self._permutation)
        state_payload = _runtime_state(self._native_runtime)
        parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state_payload["model_parameter_values"]
        )
        normalization = _decimal(
            state_payload["normalization_factor"], "runtime normalization"
        )
        self._load_evaluators()
        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            evaluation_points = (
                points
                if self._lc_replay is None
                else tuple(
                    _apply_lc_replay_input_mapping(point, entry.input_mapping)
                    for entry in self._lc_replay.entries
                    for point in points
                )
            )
            if self._helicity_plan is None:
                amplitudes = tuple(
                    self._evaluate_point(point, parameters, working_precision)
                    for point in evaluation_points
                )
                values, helicity_ids, color_ids = _reduce_resolved(
                    amplitudes,
                    self._execution,
                    self._physics,
                    normalization,
                    helicities if self._lc_replay is None else None,
                    color_flows if self._lc_replay is None else None,
                )
            else:
                values, helicity_ids, color_ids = self._evaluate_helicity_quotient(
                    evaluation_points,
                    parameters,
                    normalization,
                    working_precision,
                    helicities if self._lc_replay is None else None,
                    color_flows if self._lc_replay is None else None,
                )
            if self._lc_replay is not None:
                values, helicity_ids, color_ids = _apply_lc_replay_resolved(
                    values,
                    self._lc_replay,
                    len(points),
                    helicity_ids,
                    color_ids,
                    helicities,
                    color_flows,
                )
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            values = tuple(
                tuple(tuple(+entry for entry in colors) for colors in helicities)
                for helicities in values
            )
        return ResolvedEvaluation(
            values=values,
            helicity_ids=helicity_ids,
            color_ids=color_ids,
            color_accuracy=cast(Any, str(self._physics["color_accuracy"])),
        )

    def _load_evaluators(self) -> None:
        if self._stage_evaluators is not None:
            return
        compiled = self._execution.get("compiled")
        if not isinstance(compiled, Mapping):
            raise ArtifactError("execution metadata has no compiled evaluator set")
        stage_set = compiled.get("stage_evaluators")
        if not isinstance(stage_set, Mapping):
            raise CompatibilityError(
                "process artifact has no materialized stage evaluators"
            )
        raw_stages = stage_set.get("stages")
        amplitude = stage_set.get("amplitude_stage")
        if isinstance(raw_stages, str | bytes) or not isinstance(raw_stages, Sequence):
            raise ArtifactError("execution stage evaluator list is invalid")
        if not isinstance(amplitude, Mapping):
            raise ArtifactError("execution amplitude evaluator is invalid")
        self._stage_evaluators = tuple(
            _ExactEvaluator.load(
                _evaluator_manifest(stage),
                self._process_root,
                state_loader=self._load_exact_state,
            )
            for stage in raw_stages
        )
        self._amplitude_evaluator = _ExactEvaluator.load(
            _evaluator_manifest(amplitude),
            self._process_root,
            state_loader=self._load_exact_state,
        )

    def _load_exact_state(self, relative: str) -> bytes:
        logical_path = normalize_relative_path(
            f"{self._process_prefix}/{normalize_relative_path(relative)}"
        )
        return self._payloads.read_exact_state(
            logical_path,
            process_id=self._representative_id,
        )

    def _evaluate_point(
        self,
        point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
        model_parameters: tuple[Decimal, ...],
        precision: int,
        source_states: Sequence[_ExactRuntimeSourceState | None] | None = None,
    ) -> tuple[_ComplexDecimal, ...]:
        runtime_schema = cast(Mapping[str, object], self._execution["runtime_schema"])
        layout = cast(Mapping[str, object], runtime_schema["parameter_layout"])
        value_count = _json_integer(layout["value_component_count"])
        momentum_count = _json_integer(layout["momentum_parameter_count"])
        model_start = value_count + momentum_count
        parameter_count = max(
            _json_integer(layout["parameter_count_if_flattened"]),
            model_start + len(model_parameters),
        )
        state = [_complex_zero() for _ in range(parameter_count)]
        if source_states is None:
            _fill_sources(state, point, runtime_schema, model_parameters)
        else:
            _fill_sources_with_states(
                state,
                point,
                runtime_schema,
                model_parameters,
                source_states,
            )
        _fill_momenta(state, point, runtime_schema)
        for index, value in enumerate(model_parameters):
            state[model_start + index] = (value, _ZERO)

        stage_set = cast(
            Mapping[str, object],
            cast(Mapping[str, object], self._execution["compiled"])["stage_evaluators"],
        )
        raw_stages = cast(Sequence[object], stage_set["stages"])
        assert self._stage_evaluators is not None
        for raw_stage, evaluator in zip(
            raw_stages, self._stage_evaluators, strict=True
        ):
            stage = cast(Mapping[str, object], raw_stage)
            inputs = _pack_stage_inputs(state, stage)
            outputs = evaluator.evaluate(inputs, precision)
            _assign_stage_outputs(state, outputs, stage)

        amplitude = cast(Mapping[str, object], stage_set["amplitude_stage"])
        assert self._amplitude_evaluator is not None
        return self._amplitude_evaluator.evaluate(
            _pack_stage_inputs(state, amplitude), precision
        )

    def _evaluate_helicity_quotient(
        self,
        points: Sequence[tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]],
        model_parameters: tuple[Decimal, ...],
        normalization: Decimal,
        precision: int,
        selected_helicities: Sequence[str] | None,
        selected_colors: Sequence[str] | None,
    ) -> tuple[
        tuple[tuple[tuple[Decimal, ...], ...], ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        assert self._helicity_plan is not None
        helicities = cast(Sequence[Mapping[str, object]], self._physics["helicities"])
        colors = cast(Sequence[Mapping[str, object]], self._physics["color_components"])
        helicity_ids = tuple(str(item["id"]) for item in helicities)
        color_ids = tuple(str(item["id"]) for item in colors)
        selected_h = _selected_indices(helicity_ids, selected_helicities, "helicity")
        selected_c = _selected_indices(color_ids, selected_colors, "color component")
        if str(self._physics["color_accuracy"]) != "lc" and selected_colors is not None:
            raise EvaluationError(
                "LC color-flow selection is unavailable for NLC/full artifacts"
            )

        requested_helicities = set(selected_h)
        full_points: list[tuple[tuple[Decimal, ...], ...]] = []
        for point in points:
            full = [[_ZERO for _color in color_ids] for _helicity in helicity_ids]
            for schedule in self._helicity_plan.schedules:
                helicity_index = schedule.physical_helicity_index
                if (
                    helicity_index not in requested_helicities
                    or schedule.structural_zero
                ):
                    continue
                amplitudes = self._evaluate_point(
                    point,
                    model_parameters,
                    precision,
                    schedule.source_states,
                )
                full[helicity_index] = list(
                    _reduce_materialized_helicity(
                        amplitudes,
                        self._execution,
                        self._physics,
                        normalization,
                        helicity_index,
                        schedule.root_factors,
                    )
                )
            full_points.append(
                tuple(tuple(full[h][c] for c in selected_c) for h in selected_h)
            )
        return (
            tuple(full_points),
            tuple(helicity_ids[index] for index in selected_h),
            tuple(color_ids[index] for index in selected_c),
        )


def _selected_process(
    processes: Sequence[Mapping[str, object]],
    selected_id: str,
) -> tuple[Mapping[str, object], tuple[int, ...] | None]:
    for process in processes:
        if process["id"] == selected_id:
            return process, None
        for raw_alias in cast(Sequence[Mapping[str, object]], process["aliases"]):
            if raw_alias["id"] == selected_id:
                return process, tuple(
                    _json_integer(value)
                    for value in cast(
                        Sequence[object], raw_alias["external_permutation"]
                    )
                )
    raise ArtifactError(f"selected process {selected_id!r} is absent from artifact")


def _exact_helicity_plan(
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    public_permutation: tuple[int, ...] | None,
) -> _ExactHelicityPlan | None:
    try:
        return _parse_exact_helicity_plan(execution, physics, public_permutation)
    except (ArtifactError, CompatibilityError, EvaluationError):
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError(
            f"malformed helicity recurrence materialization: {exc}"
        ) from exc


def _parse_exact_helicity_plan(
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    public_permutation: tuple[int, ...] | None,
) -> _ExactHelicityPlan | None:
    runtime_schema = execution.get("runtime_schema")
    if not isinstance(runtime_schema, Mapping):
        raise ArtifactError("execution metadata has no runtime schema")
    recurrence = runtime_schema.get("helicity_recurrence")
    if recurrence is None:
        return None
    if not isinstance(recurrence, Mapping):
        raise ArtifactError("helicity recurrence metadata is not an object")
    materialization = recurrence.get("materialization")
    if materialization is None:
        # Pre-quotient artifacts may carry proof metadata while retaining the
        # complete DAG. Their exact execution remains unchanged.
        return None
    if not isinstance(materialization, Mapping):
        raise ArtifactError("helicity recurrence materialization is not an object")
    if _json_integer(recurrence.get("contract_version", 0)) != 1:
        raise CompatibilityError("unsupported helicity recurrence contract version")
    if (
        materialization.get("kind") != "pyamplicol-helicity-recurrence-materialization"
        or _json_integer(materialization.get("contract_version", 0)) != 1
    ):
        raise CompatibilityError(
            "unsupported helicity recurrence materialization contract"
        )

    helicities = physics.get("helicities")
    particles = physics.get("external_particles")
    if (
        isinstance(helicities, str | bytes)
        or not isinstance(helicities, Sequence)
        or isinstance(particles, str | bytes)
        or not isinstance(particles, Sequence)
        or not helicities
    ):
        raise ArtifactError("helicity recurrence requires physical helicity metadata")
    physical = cast(Sequence[Mapping[str, object]], helicities)
    external_count = len(particles)
    physical_by_values: dict[tuple[int, ...], int] = {}
    for index, helicity in enumerate(physical):
        values = _integer_vector(
            helicity.get("values"),
            external_count,
            "physical helicity vector",
        )
        if values in physical_by_values:
            raise ArtifactError("physical helicity metadata contains duplicate vectors")
        if not isinstance(helicity.get("structural_zero"), bool):
            raise ArtifactError("physical helicity structural-zero flag is not boolean")
        physical_by_values[values] = index
    if public_permutation is not None and (
        len(public_permutation) != external_count
        or set(public_permutation) != set(range(external_count))
    ):
        raise ArtifactError(
            "process alias has an invalid public-label permutation for "
            "helicity recurrence"
        )

    raw_domains = recurrence.get("selector_domains")
    if isinstance(raw_domains, str | bytes) or not isinstance(raw_domains, Sequence):
        raise ArtifactError("helicity recurrence selector domains are invalid")
    domains: dict[int, tuple[bool, frozenset[tuple[int, int]]]] = {}
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, Mapping):
            raise ArtifactError("helicity recurrence selector domain is not an object")
        domain_id = _strict_nonnegative_integer(
            raw_domain.get("id"), "helicity selector domain ID"
        )
        complete = raw_domain.get("complete")
        if not isinstance(complete, bool):
            raise ArtifactError("helicity selector domain completeness is not boolean")
        raw_states = raw_domain.get("source_states")
        if isinstance(raw_states, str | bytes) or not isinstance(raw_states, Sequence):
            raise ArtifactError("helicity selector domain source states are invalid")
        domain_states: set[tuple[int, int]] = set()
        for raw_state in raw_states:
            if not isinstance(raw_state, Mapping):
                raise ArtifactError("helicity selector source state is not an object")
            label = _strict_nonnegative_integer(
                raw_state.get("external_label"), "helicity source external label"
            )
            if not 1 <= label <= external_count:
                raise ArtifactError("helicity source external label is out of range")
            state = (label - 1, _json_integer(raw_state.get("helicity")))
            if state in domain_states or any(
                existing[0] == state[0] for existing in domain_states
            ):
                raise ArtifactError(
                    "helicity selector domain repeats an external label"
                )
            domain_states.add(state)
        if complete and len(domain_states) != external_count:
            raise ArtifactError("complete helicity selector domain is incomplete")
        if domain_id in domains:
            raise ArtifactError("helicity recurrence repeats a selector domain ID")
        domains[domain_id] = (complete, frozenset(domain_states))

    source_fill = runtime_schema.get("source_fill")
    amplitude_stage = runtime_schema.get("amplitude_stage")
    if not isinstance(source_fill, Mapping) or not isinstance(amplitude_stage, Mapping):
        raise ArtifactError("helicity materialization has incomplete runtime metadata")
    raw_sources = source_fill.get("sources")
    raw_roots = amplitude_stage.get("roots")
    if (
        isinstance(raw_sources, str | bytes)
        or not isinstance(raw_sources, Sequence)
        or isinstance(raw_roots, str | bytes)
        or not isinstance(raw_roots, Sequence)
    ):
        raise ArtifactError("helicity materialization source/root metadata is invalid")
    sources = cast(Sequence[Mapping[str, object]], raw_sources)
    roots = cast(Sequence[Mapping[str, object]], raw_roots)
    if _json_integer(materialization.get("materialized_root_count", -1)) != len(roots):
        raise ArtifactError("helicity materialization root count is inconsistent")
    for index, root in enumerate(roots):
        if (
            _strict_nonnegative_integer(root.get("root_id"), "amplitude root ID")
            != index
            or _strict_nonnegative_integer(
                root.get("output_index"), "amplitude root output index"
            )
            != index
        ):
            raise ArtifactError(
                "helicity materialization requires contiguous root/output identifiers"
            )

    source_routes = _exact_source_routes(materialization, domains, sources)
    amplitude_routes = _exact_amplitude_routes(materialization, domains, len(roots))
    raw_schedules = materialization.get("selector_schedules")
    if isinstance(raw_schedules, str | bytes) or not isinstance(
        raw_schedules, Sequence
    ):
        raise ArtifactError("helicity materialization selector schedules are invalid")
    schedules_by_physical: dict[int, _ExactHelicitySchedule] = {}
    for raw_schedule in raw_schedules:
        if not isinstance(raw_schedule, Mapping):
            raise ArtifactError("helicity materialization schedule is not an object")
        domain_id = _strict_nonnegative_integer(
            raw_schedule.get("selector_domain_id"), "helicity schedule domain ID"
        )
        domain = domains.get(domain_id)
        if domain is None or not domain[0]:
            raise ArtifactError(
                "helicity schedule does not reference a complete domain"
            )
        representative_values = _complete_domain_values(domain[1], external_count)
        public_values = (
            representative_values
            if public_permutation is None
            else _permute_helicity_to_public_alias(
                representative_values, public_permutation
            )
        )
        physical_index = physical_by_values.get(public_values)
        if physical_index is None:
            raise ArtifactError(
                f"helicity selector domain {public_values!r} has no physical entry"
            )
        if physical_index in schedules_by_physical:
            raise ArtifactError("physical helicity has multiple recurrence schedules")
        structural_zero = raw_schedule.get("structural_zero")
        if not isinstance(structural_zero, bool):
            raise ArtifactError("helicity schedule structural-zero flag is not boolean")
        if bool(physical[physical_index].get("structural_zero")) != structural_zero:
            raise ArtifactError(
                "helicity schedule structural-zero flag disagrees with physics metadata"
            )
        active_roots = frozenset(
            _strict_integer_sequence(
                raw_schedule.get("active_root_ids"),
                len(roots),
                "helicity schedule active roots",
            )
        )
        active_currents = _strict_integer_sequence(
            raw_schedule.get("active_current_ids"),
            _json_integer(materialization.get("materialized_current_count", -1)),
            "helicity schedule active currents",
        )
        if structural_zero and (active_currents or active_roots):
            raise ArtifactError(
                "structural-zero helicity schedule has an active closure"
            )
        schedule_source_states = (
            ()
            if structural_zero
            else _source_states_for_active_closure(
                sources,
                source_routes,
                active_currents,
                domain[1],
            )
        )
        factors: list[_ComplexDecimal | None] = [None] * len(roots)
        for root_id, route_domain_ids, factor in amplitude_routes:
            if domain_id not in route_domain_ids:
                continue
            if factors[root_id] is not None:
                raise ArtifactError(
                    "helicity schedule routes one materialized root more than once"
                )
            factors[root_id] = factor
        routed = frozenset(
            index for index, factor in enumerate(factors) if factor is not None
        )
        if routed != active_roots:
            raise ArtifactError(
                "helicity schedule amplitude routes do not match active roots"
            )
        schedules_by_physical[physical_index] = _ExactHelicitySchedule(
            physical_helicity_index=physical_index,
            selector_domain_id=domain_id,
            structural_zero=structural_zero,
            source_states=schedule_source_states,
            root_factors=tuple(factors),
        )
    if set(schedules_by_physical) != set(range(len(physical))):
        raise ArtifactError(
            "helicity materialization does not cover every physical helicity"
        )
    return _ExactHelicityPlan(
        schedules=tuple(schedules_by_physical[index] for index in range(len(physical)))
    )


def _exact_source_routes(
    materialization: Mapping[str, object],
    domains: Mapping[int, tuple[bool, frozenset[tuple[int, int]]]],
    sources: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, int, int, int, _ExactRuntimeSourceState], ...]:
    raw_routes = materialization.get("source_routes")
    if isinstance(raw_routes, str | bytes) or not isinstance(raw_routes, Sequence):
        raise ArtifactError("helicity materialization source routes are invalid")
    source_by_current = {
        _strict_nonnegative_integer(
            source.get("current_id"), "source current ID"
        ): source
        for source in sources
    }
    if len(source_by_current) != len(sources):
        raise ArtifactError("runtime source currents are not unique")
    parsed = []
    for raw_route in raw_routes:
        if not isinstance(raw_route, Mapping):
            raise ArtifactError(
                "helicity materialization source route is not an object"
            )
        current_id = _strict_nonnegative_integer(
            raw_route.get("materialized_current_id"), "source-route current ID"
        )
        source = source_by_current.get(current_id)
        if source is None:
            raise ArtifactError(
                "source route references an unknown materialized current"
            )
        external_label = _strict_nonnegative_integer(
            raw_route.get("external_label"), "source-route external label"
        )
        if external_label != _json_integer(source.get("leg_label")):
            raise ArtifactError("source route external label disagrees with its source")
        domain_id = _strict_nonnegative_integer(
            raw_route.get("selector_domain_id"), "source-route selector domain"
        )
        if domain_id not in domains:
            raise ArtifactError("source route references an unknown selector domain")
        declared_index = _strict_nonnegative_integer(
            raw_route.get("declared_state_index"), "source-route declared state"
        )
        helicity, chirality, spin_state = _transformed_declared_source_state(
            source, declared_index
        )
        if (
            _json_integer(raw_route.get("helicity")) != helicity
            or _json_integer(raw_route.get("chirality")) != chirality
            or _json_integer(raw_route.get("spin_state")) != spin_state
        ):
            raise ArtifactError(
                "source route disagrees with its declared transformed state"
            )
        factor = _complex_decimal_pair(raw_route.get("factor"), "source-route factor")
        if (external_label - 1, helicity) not in domains[domain_id][1]:
            raise ArtifactError(
                "source route selector domain does not contain its source state"
            )
        parsed.append(
            (
                current_id,
                external_label - 1,
                helicity,
                domain_id,
                _ExactRuntimeSourceState(helicity, chirality, spin_state, factor),
            )
        )
    return tuple(parsed)


def _exact_amplitude_routes(
    materialization: Mapping[str, object],
    domains: Mapping[int, tuple[bool, frozenset[tuple[int, int]]]],
    root_count: int,
) -> tuple[tuple[int, frozenset[int], _ComplexDecimal], ...]:
    raw_routes = materialization.get("amplitude_routes")
    if isinstance(raw_routes, str | bytes) or not isinstance(raw_routes, Sequence):
        raise ArtifactError("helicity materialization amplitude routes are invalid")
    parsed = []
    for raw_route in raw_routes:
        if not isinstance(raw_route, Mapping):
            raise ArtifactError(
                "helicity materialization amplitude route is not an object"
            )
        root_id = _strict_nonnegative_integer(
            raw_route.get("materialized_root_id"), "amplitude-route root ID"
        )
        if root_id >= root_count:
            raise ArtifactError("amplitude route references an unknown root")
        domain_ids = frozenset(
            _strict_integer_sequence(
                raw_route.get("selector_domain_ids"),
                max(domains, default=-1) + 1,
                "amplitude-route selector domains",
            )
        )
        if not domain_ids or any(domain_id not in domains for domain_id in domain_ids):
            raise ArtifactError("amplitude route references an unknown selector domain")
        parsed.append(
            (
                root_id,
                domain_ids,
                _complex_decimal_pair(
                    raw_route.get("factor"), "amplitude-route factor"
                ),
            )
        )
    return tuple(parsed)


def _source_states_for_active_closure(
    sources: Sequence[Mapping[str, object]],
    routes: Sequence[tuple[int, int, int, int, _ExactRuntimeSourceState]],
    active_current_ids: Sequence[int],
    complete_states: frozenset[tuple[int, int]],
) -> tuple[_ExactRuntimeSourceState | None, ...]:
    active = frozenset(active_current_ids)
    active_labels = {
        _json_integer(source.get("leg_label")) - 1
        for source in sources
        if _json_integer(source.get("current_id")) in active
    }
    expected_labels = {external for external, _helicity in complete_states}
    if active_labels != expected_labels:
        raise ArtifactError(
            "helicity selector source-label coverage disagrees with its complete domain"
        )
    result: list[_ExactRuntimeSourceState | None] = []
    state_by_external = dict(complete_states)
    for source in sources:
        current_id = _json_integer(source.get("current_id"))
        if current_id not in active:
            result.append(None)
            continue
        external = _json_integer(source.get("leg_label")) - 1
        try:
            selected_helicity = state_by_external[external]
        except KeyError as exc:
            raise ArtifactError(
                "active materialized source has no state in its complete domain"
            ) from exc
        matching = [
            state
            for route_current, route_external, helicity, _domain_id, state in routes
            if route_current == current_id
            and route_external == external
            and helicity == selected_helicity
        ]
        if len(matching) > 1:
            raise ArtifactError(
                "active materialized source has multiple routes for its "
                "selected helicity"
            )
        result.append(
            matching[0]
            if matching
            else _declared_source_state_for_helicity(source, selected_helicity)
        )
    return tuple(result)


def _declared_source_state_for_helicity(
    source: Mapping[str, object], helicity: int
) -> _ExactRuntimeSourceState:
    source_ir, _identity, _crossing = _validated_source_ir(source)
    raw_states = source_ir.get("states")
    if isinstance(raw_states, str | bytes) or not isinstance(raw_states, Sequence):
        raise ArtifactError("source IR declared states are invalid")
    matching = []
    for declared_index in range(len(raw_states)):
        runtime_helicity, chirality, spin_state = _transformed_declared_source_state(
            source, declared_index
        )
        if runtime_helicity == helicity:
            matching.append(
                _ExactRuntimeSourceState(
                    helicity=runtime_helicity,
                    chirality=chirality,
                    spin_state=spin_state,
                    factor=(_ONE, _ZERO),
                )
            )
    if len(matching) != 1:
        raise ArtifactError(
            "active materialized source does not declare exactly one state "
            f"for helicity {helicity}"
        )
    return matching[0]


def _transformed_declared_source_state(
    source: Mapping[str, object], declared_index: int
) -> tuple[int, int, int]:
    source_ir, _identity, crossing = _validated_source_ir(source)
    raw_states = source_ir.get("states")
    if isinstance(raw_states, str | bytes) or not isinstance(raw_states, Sequence):
        raise ArtifactError("source IR declared states are invalid")
    try:
        state = raw_states[declared_index]
    except IndexError as exc:
        raise ArtifactError("source route has a stale declared state index") from exc
    if not isinstance(state, Mapping):
        raise ArtifactError("source IR declared state is not an object")
    return (
        _json_integer(state.get("helicity"))
        * _json_integer(crossing.get("helicity_factor")),
        _json_integer(state.get("chirality"))
        * _json_integer(crossing.get("chirality_factor")),
        _json_integer(state.get("spin_state"))
        * _json_integer(crossing.get("spin_state_factor")),
    )


def _complete_domain_values(
    states: frozenset[tuple[int, int]], external_count: int
) -> tuple[int, ...]:
    values: list[int | None] = [None] * external_count
    for external_index, helicity in states:
        values[external_index] = helicity
    if any(value is None for value in values):
        raise ArtifactError("complete helicity selector domain is incomplete")
    return tuple(cast(int, value) for value in values)


def _permute_helicity_to_public_alias(
    representative: tuple[int, ...], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    public = [0] * len(representative)
    for representative_index, public_index in enumerate(permutation):
        public[public_index] = representative[representative_index]
    return tuple(public)


def _integer_vector(value: object, length: int, context: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} is invalid")
    result = tuple(_json_integer(item) for item in value)
    if len(result) != length:
        raise ArtifactError(f"{context} has an inconsistent length")
    return result


def _external_label_word(
    value: object,
    external_count: int,
    context: str,
    *,
    expected_labels: frozenset[int] | None = None,
) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} is invalid")
    result = tuple(_json_integer(item) for item in value)
    labels = frozenset(result)
    if (
        not result
        or len(labels) != len(result)
        or any(label < 1 or label > external_count for label in result)
        or (expected_labels is not None and labels != expected_labels)
    ):
        raise ArtifactError(f"{context} has inconsistent external labels")
    return result


def _strict_nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{context} is not a non-negative integer")
    return value


def _strict_integer_sequence(
    value: object, upper_bound: int, context: str
) -> tuple[int, ...]:
    if (
        upper_bound < 0
        or isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
    ):
        raise ArtifactError(f"{context} are invalid")
    result = tuple(
        _strict_nonnegative_integer(item, f"{context} entry") for item in value
    )
    if len(set(result)) != len(result) or any(item >= upper_bound for item in result):
        raise ArtifactError(f"{context} are invalid")
    return result


def _complex_decimal_pair(value: object, context: str) -> _ComplexDecimal:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ArtifactError(f"{context} is not a complex pair")
    return (_decimal(value[0], context), _decimal(value[1], context))


def _lc_replay_plan(
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    public_permutation: tuple[int, ...] | None,
) -> _LcReplayPlan | None:
    compiled = execution.get("compiled")
    compiled_replay = (
        compiled.get("lc_topology_replay")
        if isinstance(compiled, Mapping)
        else None
    )
    eager_replay = execution.get("lc_topology_replay")
    if (
        compiled_replay is not None
        and eager_replay is not None
        and compiled_replay != eager_replay
    ):
        raise ArtifactError(
            "LC topology replay metadata disagrees between execution lanes"
        )
    raw_replay = compiled_replay if compiled_replay is not None else eager_replay
    if raw_replay is None:
        return None
    if not isinstance(raw_replay, Mapping):
        raise ArtifactError("LC topology replay metadata is not an object")
    if not bool(raw_replay.get("enabled")):
        return None
    if raw_replay.get("mode") != "external-label-permutation":
        raise CompatibilityError(
            f"unsupported LC topology replay mode {raw_replay.get('mode')!r}"
        )
    particles = physics.get("external_particles")
    if isinstance(particles, str | bytes) or not isinstance(particles, Sequence):
        raise ArtifactError("physics metadata has no external particles")
    external_count = len(particles)
    if public_permutation is not None and (
        len(public_permutation) != external_count
        or set(public_permutation) != set(range(external_count))
    ):
        raise ArtifactError(
            "process alias has an invalid public-label permutation for LC "
            "topology replay"
        )
    contract_version = _json_integer(raw_replay.get("contract_version", 1))
    if contract_version == 2:
        return _lc_replay_plan_v2(
            execution,
            physics,
            raw_replay,
            public_permutation,
            external_count,
        )
    if contract_version != 1:
        raise CompatibilityError(
            f"unsupported LC topology replay contract version {contract_version}"
        )

    raw_groups = raw_replay.get("groups")
    if isinstance(raw_groups, str | bytes) or not isinstance(raw_groups, Sequence):
        raise ArtifactError("LC topology replay group list is invalid")
    parsed: list[
        tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], Decimal]
    ] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            raise ArtifactError("LC topology replay group is not an object")
        raw_permutations = raw_group.get("sector_permutations")
        if isinstance(raw_permutations, str | bytes) or not isinstance(
            raw_permutations, Sequence
        ):
            raise ArtifactError("LC topology replay sector permutations are invalid")
        for raw_permutation in raw_permutations:
            if not isinstance(raw_permutation, Mapping):
                raise ArtifactError(
                    "LC topology replay sector permutation is not an object"
                )
            raw_labels = raw_permutation.get("label_permutation", ())
            if isinstance(raw_labels, str | bytes) or not isinstance(
                raw_labels, Sequence
            ):
                raise ArtifactError("LC topology replay label permutation is invalid")
            mapping: list[tuple[int, int]] = []
            for raw_label in raw_labels:
                if not isinstance(raw_label, Mapping):
                    raise ArtifactError(
                        "LC topology replay label permutation entry is not an object"
                    )
                representative = _json_integer(raw_label["representative_label"]) - 1
                sector = _json_integer(raw_label["sector_label"]) - 1
                mapping.append((representative, sector))
            input_mapping = tuple(mapping)
            _validate_lc_public_mapping(input_mapping, external_count)
            public_mapping = (
                input_mapping
                if public_permutation is None
                else tuple(
                    (public_permutation[representative], public_permutation[sector])
                    for representative, sector in input_mapping
                )
            )
            replay_weight = _decimal(
                raw_permutation.get("weight", 1), "LC topology replay weight"
            )
            if replay_weight <= 0:
                raise ArtifactError(
                    "LC topology replay weights must be positive finite numbers"
                )
            parsed.append((input_mapping, public_mapping, replay_weight))
    declared_count = _json_integer(raw_replay.get("replayed_sector_count", 0))
    if declared_count != len(parsed) or not parsed:
        raise ArtifactError(
            "LC topology replay sector count does not match its permutations"
        )

    helicities = cast(Sequence[Mapping[str, object]], physics["helicities"])
    colors = cast(Sequence[Mapping[str, object]], physics["color_components"])
    if str(physics.get("color_accuracy")) != "lc":
        raise ArtifactError("LC topology replay requires LC resolved physics metadata")
    helicity_ids = tuple(str(item["id"]) for item in helicities)
    color_ids = tuple(str(item["id"]) for item in colors)
    helicity_index = {
        identifier: index for index, identifier in enumerate(helicity_ids)
    }
    color_index = {identifier: index for index, identifier in enumerate(color_ids)}
    helicity_by_values: dict[tuple[int, ...], int] = {}
    for index, record in enumerate(helicities):
        values = tuple(
            _json_integer(value) for value in cast(Sequence[object], record["values"])
        )
        if values in helicity_by_values:
            raise ArtifactError("resolved physics contains duplicate helicity vectors")
        helicity_by_values[values] = index
    color_by_word: dict[tuple[int, ...], int] = {}
    for index, record in enumerate(colors):
        if record.get("kind") != "lc-flow":
            raise ArtifactError(
                "LC topology replay encountered a contracted color component"
            )
        word = tuple(
            _json_integer(value) for value in cast(Sequence[object], record["word"])
        )
        if word in color_by_word:
            raise ArtifactError("resolved physics contains duplicate LC flow words")
        color_by_word[word] = index

    reduction = physics.get("reduction")
    if not isinstance(reduction, Mapping):
        raise ArtifactError("resolved physics has no reduction metadata")
    raw_reductions = reduction.get("groups")
    if isinstance(raw_reductions, str | bytes) or not isinstance(
        raw_reductions, Sequence
    ):
        raise ArtifactError("resolved physics reduction groups are invalid")
    relevant_helicities: set[int] = set()
    relevant_colors: set[int] = set()
    for raw_reduction in raw_reductions:
        if not isinstance(raw_reduction, Mapping):
            raise ArtifactError("resolved physics reduction group is invalid")
        for raw_id in cast(Sequence[object], raw_reduction["physical_helicity_ids"]):
            identifier = str(raw_id)
            if identifier not in helicity_index:
                raise ArtifactError(
                    f"reduction references unknown helicity {identifier!r}"
                )
            relevant_helicities.add(helicity_index[identifier])
        for raw_id in cast(Sequence[object], raw_reduction["physical_color_ids"]):
            identifier = str(raw_id)
            if identifier not in color_index:
                raise ArtifactError(
                    f"reduction references unknown color component {identifier!r}"
                )
            relevant_colors.add(color_index[identifier])
    if not relevant_helicities or not relevant_colors:
        raise ArtifactError("LC topology replay has no materialized resolved members")

    entries: list[_LcReplayEntry] = []
    color_count = len(colors)
    for input_mapping, public_mapping, replay_weight in parsed:
        helicity_targets: dict[int, int] = {}
        for source_index in sorted(relevant_helicities):
            source = helicities[source_index]
            source_values = tuple(
                _json_integer(value)
                for value in cast(Sequence[object], source["values"])
            )
            target_values = _permute_lc_helicity(source_values, public_mapping)
            target_index = helicity_by_values.get(target_values)
            if target_index is None:
                raise CompatibilityError(
                    "resolved physics is missing replayed helicity vector "
                    f"{target_values!r}; regenerate the artifact with complete "
                    "topology-replay reductions"
                )
            source_coefficient = _decimal(source["coefficient"], "helicity coefficient")
            target_coefficient = _decimal(
                helicities[target_index]["coefficient"], "helicity coefficient"
            )
            if source_coefficient != target_coefficient:
                raise ArtifactError(
                    "LC topology replay maps helicities with different reduction "
                    "coefficients"
                )
            helicity_targets[source_index] = target_index

        color_targets: dict[int, tuple[tuple[int, Decimal], ...]] = {}
        for source_index in sorted(relevant_colors):
            source_word = tuple(
                _json_integer(value)
                for value in cast(Sequence[object], colors[source_index]["word"])
            )
            target_word = _permute_lc_color_word(source_word, public_mapping)
            target_words = _lc_replay_public_words(target_word, replay_weight)
            coefficients: dict[int, Decimal] = {}
            for word in target_words:
                target_index = color_by_word.get(word)
                if target_index is None:
                    raise CompatibilityError(
                        "resolved physics is missing replayed LC flow word "
                        f"{word!r}; regenerate the artifact with "
                        "replay-to-public-flow reductions"
                    )
                coefficient = _decimal(
                    colors[target_index].get("coefficient", 1),
                    "color coefficient",
                )
                if coefficient <= 0:
                    raise ArtifactError(
                        "replayed LC flow has no positive reduction coefficient"
                    )
                coefficients.setdefault(target_index, coefficient)
            total = sum(coefficients.values(), _ZERO)
            if total <= 0:
                raise ArtifactError(
                    "LC topology replay has no positive public-flow reduction weight"
                )
            color_targets[source_index] = tuple(
                (target_index, replay_weight * coefficient / total)
                for target_index, coefficient in sorted(coefficients.items())
            )

        routes = tuple(
            _LcReplayRoute(
                source_index=source_helicity * color_count + source_color,
                target_index=target_helicity * color_count + target_color,
                weight=weight,
            )
            for source_helicity, target_helicity in sorted(helicity_targets.items())
            for source_color in sorted(relevant_colors)
            for target_color, weight in color_targets[source_color]
        )
        entries.append(_LcReplayEntry(input_mapping=input_mapping, routes=routes))
    return _LcReplayPlan(
        entries=tuple(entries),
        helicity_count=len(helicities),
        color_count=color_count,
    )


def _lc_replay_plan_v2(
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    replay: Mapping[str, object],
    public_permutation: tuple[int, ...] | None,
    external_count: int,
) -> _LcReplayPlan:
    if str(physics.get("color_accuracy")) != "lc":
        raise ArtifactError("LC topology replay requires LC resolved physics metadata")
    helicities = _mapping_records(physics, "helicities", "resolved physics helicities")
    colors = _mapping_records(
        physics, "color_components", "resolved physics color components"
    )
    helicity_ids = tuple(str(record.get("id")) for record in helicities)
    color_ids = tuple(str(record.get("id")) for record in colors)
    if len(set(helicity_ids)) != len(helicity_ids):
        raise ArtifactError("resolved physics contains duplicate helicity IDs")
    if len(set(color_ids)) != len(color_ids):
        raise ArtifactError("resolved physics contains duplicate color IDs")
    helicity_index = {
        identifier: index for index, identifier in enumerate(helicity_ids)
    }
    color_index = {identifier: index for index, identifier in enumerate(color_ids)}
    helicity_by_values: dict[tuple[int, ...], int] = {}
    for index, record in enumerate(helicities):
        values = _integer_vector(
            record.get("values"), external_count, "resolved helicity vector"
        )
        if values in helicity_by_values:
            raise ArtifactError("resolved physics contains duplicate helicity vectors")
        helicity_by_values[values] = index
    color_by_word: dict[tuple[int, ...], int] = {}
    color_labels: frozenset[int] | None = None
    for index, record in enumerate(colors):
        if record.get("kind") != "lc-flow":
            raise ArtifactError(
                "LC topology replay encountered a contracted color component"
            )
        word = _external_label_word(
            record.get("word"),
            external_count,
            "LC flow word",
            expected_labels=color_labels,
        )
        if color_labels is None:
            color_labels = frozenset(word)
        if word in color_by_word:
            raise ArtifactError("resolved physics contains duplicate LC flow words")
        color_by_word[word] = index

    reductions = _lc_reductions(physics, helicity_index, color_index)
    relevant_helicities = {
        helicity_index[identifier]
        for reduction in reductions.values()
        for identifier in _string_members(
            reduction, "physical_helicity_ids", "reduction physical helicities"
        )
    }
    if not relevant_helicities:
        raise ArtifactError("LC topology replay has no materialized helicities")
    materialized = _lc_materialized_sectors(
        execution,
        colors,
        color_index,
        reductions,
    )

    raw_groups = _mapping_records(replay, "groups", "LC topology replay groups")
    routes_by_mapping: dict[
        tuple[tuple[int, int], ...],
        tuple[tuple[tuple[int, int], ...], list[_LcReplaySectorRoute]],
    ] = {}
    replayed_sector_ids: set[int] = set()
    group_materialized_sector_ids: set[int] = set()
    for group in raw_groups:
        _validate_lc_replay_proof(group.get("proof"))
        representative_sector_id = _strict_nonnegative_integer(
            group.get("representative_sector_id"), "LC representative sector ID"
        )
        materialized_sector_id = _strict_nonnegative_integer(
            group.get("materialized_sector_id"), "LC materialized sector ID"
        )
        if materialized_sector_id != representative_sector_id:
            raise ArtifactError(
                "LC replay materialized sector is not its representative sector"
            )
        if materialized_sector_id in group_materialized_sector_ids:
            raise ArtifactError(
                "LC replay groups reuse a materialized representative sector"
            )
        group_materialized_sector_ids.add(materialized_sector_id)
        active_sector_ids = set(
            _nonnegative_integer_members(
                group,
                "active_sector_ids",
                "LC replay active sector IDs",
            )
        )
        if representative_sector_id not in active_sector_ids:
            raise ArtifactError(
                "LC replay active sectors omit the representative sector"
            )
        if replayed_sector_ids & active_sector_ids:
            raise ArtifactError("LC replay groups overlap physical sector coverage")
        replayed_sector_ids.update(active_sector_ids)
        raw_permutations = _mapping_records(
            group,
            "sector_permutations",
            "LC topology replay sector permutations",
        )
        seen_sector_ids: set[int] = set()
        for permutation in raw_permutations:
            sector_id = _strict_nonnegative_integer(
                permutation.get("sector_id"), "LC replay physical sector ID"
            )
            if sector_id not in active_sector_ids or sector_id in seen_sector_ids:
                raise ArtifactError(
                    "LC replay permutations do not uniquely cover active sectors"
                )
            seen_sector_ids.add(sector_id)
            mapping = _lc_v2_label_mapping(
                permutation.get("label_permutation"), external_count
            )
            public_mapping = (
                mapping
                if public_permutation is None
                else tuple(
                    (public_permutation[representative], public_permutation[sector])
                    for representative, sector in mapping
                )
            )
            weight = _decimal(permutation.get("weight"), "LC replay sector weight")
            sign = _json_integer(permutation.get("sign"))
            if weight <= 0 or sign not in {-1, 1}:
                raise ArtifactError(
                    "LC replay sector weight/sign must be positive and +/-1"
                )
            factor = _complex_decimal_pair(
                permutation.get("factor"), "LC replay signed factor"
            )
            if factor != (weight * sign, _ZERO):
                raise ArtifactError(
                    "LC replay signed factor does not match its weight and sign"
                )
            stored_public, mapping_routes = routes_by_mapping.setdefault(
                mapping, (public_mapping, [])
            )
            if stored_public != public_mapping:
                raise ArtifactError("LC replay has inconsistent public label mappings")
            mapping_routes.append(
                _LcReplaySectorRoute(
                    physical_sector_id=sector_id,
                    materialized_sector_id=materialized_sector_id,
                    reduction_weight=weight,
                    residual=False,
                )
            )
        if seen_sector_ids != active_sector_ids:
            raise ArtifactError(
                "LC replay permutations do not exactly cover active sectors"
            )

    declared_replayed = _strict_nonnegative_integer(
        replay.get("replayed_sector_count"), "LC replayed sector count"
    )
    if declared_replayed != len(replayed_sector_ids):
        raise ArtifactError("LC replayed sector count is inconsistent")
    residual_sector_ids = set(
        _nonnegative_integer_members(
            replay,
            "residual_sector_ids",
            "LC replay residual sector IDs",
        )
    )
    if replayed_sector_ids & residual_sector_ids:
        raise ArtifactError("LC replay residual and replayed sectors overlap")
    identity_public: tuple[tuple[int, int], ...] = ()
    stored_public, identity_routes = routes_by_mapping.setdefault(
        (), (identity_public, [])
    )
    if stored_public != identity_public:
        raise ArtifactError("LC replay identity mapping is inconsistent")
    for sector_id in sorted(residual_sector_ids):
        sector = materialized.get(sector_id)
        if sector is None:
            raise ArtifactError(
                f"LC replay residual sector {sector_id} is not materialized"
            )
        identity_routes.append(
            _LcReplaySectorRoute(
                physical_sector_id=sector_id,
                materialized_sector_id=sector_id,
                reduction_weight=sector.reduction_weight,
                residual=True,
            )
        )

    declared_materialized = set(
        _nonnegative_integer_members(
            replay,
            "materialized_sector_ids",
            "LC replay materialized sector IDs",
        )
    )
    expected_materialized = group_materialized_sector_ids | residual_sector_ids
    if declared_materialized != expected_materialized:
        raise ArtifactError("LC replay materialized sector coverage is inconsistent")
    physical_sector_count = _strict_nonnegative_integer(
        replay.get("physical_sector_count"), "LC physical sector count"
    )
    if replayed_sector_ids | residual_sector_ids != set(range(physical_sector_count)):
        raise ArtifactError(
            "LC replay does not cover every physical sector exactly once"
        )
    if not declared_materialized.issubset(materialized):
        raise ArtifactError("LC replay declares an absent materialized sector")

    entries: list[_LcReplayEntry] = []
    target_sector_by_color: dict[int, int] = {}
    color_count = len(colors)
    for input_mapping in sorted(routes_by_mapping):
        public_mapping, sector_routes = routes_by_mapping[input_mapping]
        helicity_targets: dict[int, int] = {}
        for source_index in sorted(relevant_helicities):
            source = helicities[source_index]
            source_values = _integer_vector(
                source.get("values"), external_count, "resolved helicity vector"
            )
            target_values = _permute_lc_helicity(source_values, public_mapping)
            target_index = helicity_by_values.get(target_values)
            if target_index is None:
                raise CompatibilityError(
                    "resolved physics is missing replayed helicity vector "
                    f"{target_values!r}; regenerate the artifact with complete "
                    "topology-replay reductions"
                )
            if _decimal(source.get("coefficient"), "helicity coefficient") != _decimal(
                helicities[target_index].get("coefficient"),
                "helicity coefficient",
            ):
                raise ArtifactError(
                    "LC topology replay maps helicities with different coefficients"
                )
            helicity_targets[source_index] = target_index

        resolved_routes: list[_LcReplayRoute] = []
        for sector_route in sorted(
            sector_routes, key=lambda route: route.physical_sector_id
        ):
            source_sector = materialized.get(sector_route.materialized_sector_id)
            if source_sector is None:
                raise ArtifactError(
                    "LC replay route references an absent materialized sector"
                )
            source_index = source_sector.color_index
            source = colors[source_index]
            source_id = str(source.get("id"))
            if (
                source.get("computed") is not True
                or source.get("representative_id") != source_id
            ):
                raise ArtifactError(
                    "LC replay materialized sector is not a computed representative"
                )
            source_word = _external_label_word(
                source.get("word"),
                external_count,
                "materialized LC flow word",
                expected_labels=color_labels,
            )
            target_word = _permute_lc_color_word(source_word, public_mapping)
            target_words = _lc_replay_public_words(
                target_word, sector_route.reduction_weight
            )
            target_coefficients: dict[int, Decimal] = {}
            for word in target_words:
                target_index = color_by_word.get(word)
                if target_index is None:
                    raise CompatibilityError(
                        "resolved physics is missing replayed LC flow word "
                        f"{word!r}; regenerate the artifact with "
                        "replay-to-public-flow reductions"
                    )
                target = colors[target_index]
                if target.get("representative_id") != source_id:
                    raise ArtifactError(
                        "LC replay target flow has the wrong representative"
                    )
                previous_sector = target_sector_by_color.setdefault(
                    target_index, sector_route.physical_sector_id
                )
                if previous_sector != sector_route.physical_sector_id:
                    raise ArtifactError(
                        "LC replay physical sectors overlap a public color flow"
                    )
                coefficient = _decimal(
                    target.get("coefficient", 1), "color coefficient"
                )
                if coefficient <= 0:
                    raise ArtifactError(
                        "replayed LC flow has no positive reduction coefficient"
                    )
                target_coefficients.setdefault(target_index, coefficient)
            total = sum(target_coefficients.values(), _ZERO)
            if total <= 0:
                raise ArtifactError(
                    "LC topology replay has no positive public-flow weight"
                )
            for source_helicity, target_helicity in sorted(helicity_targets.items()):
                for target_color, coefficient in sorted(target_coefficients.items()):
                    resolved_routes.append(
                        _LcReplayRoute(
                            source_index=source_helicity * color_count + source_index,
                            target_index=target_helicity * color_count + target_color,
                            weight=(
                                sector_route.reduction_weight * coefficient / total
                            ),
                        )
                    )
        if not resolved_routes:
            raise ArtifactError("LC topology replay mapping has no resolved routes")
        entries.append(
            _LcReplayEntry(
                input_mapping=input_mapping,
                routes=tuple(resolved_routes),
            )
        )
    if set(target_sector_by_color) != set(range(color_count)):
        raise ArtifactError(
            f"LC topology replay covers {len(target_sector_by_color)} of "
            f"{color_count} public color components"
        )
    return _LcReplayPlan(
        entries=tuple(entries),
        helicity_count=len(helicities),
        color_count=color_count,
    )


def _lc_reductions(
    physics: Mapping[str, object],
    helicity_index: Mapping[str, int],
    color_index: Mapping[str, int],
) -> dict[int, Mapping[str, object]]:
    reduction = physics.get("reduction")
    if not isinstance(reduction, Mapping):
        raise ArtifactError("resolved physics has no reduction metadata")
    records = _mapping_records(reduction, "groups", "resolved physics reduction groups")
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        identifier = str(record.get("id", ""))
        if not identifier.startswith("reduction:"):
            raise ArtifactError("resolved reduction group has an invalid ID")
        try:
            group_id = int(identifier.removeprefix("reduction:"))
        except ValueError as exc:
            raise ArtifactError("resolved reduction group has an invalid ID") from exc
        if group_id < 0 or group_id in result:
            raise ArtifactError("resolved reduction group IDs are invalid")
        helicity_ids = _string_members(
            record, "physical_helicity_ids", "reduction physical helicities"
        )
        color_ids = _string_members(
            record, "physical_color_ids", "reduction physical colors"
        )
        if not helicity_ids or any(
            value not in helicity_index for value in helicity_ids
        ):
            raise ArtifactError("reduction references an unknown physical helicity")
        if not color_ids or any(value not in color_index for value in color_ids):
            raise ArtifactError("reduction references an unknown physical color")
        result[group_id] = record
    return result


def _lc_materialized_sectors(
    execution: Mapping[str, object],
    colors: Sequence[Mapping[str, object]],
    color_index: Mapping[str, int],
    reductions: Mapping[int, Mapping[str, object]],
) -> dict[int, _LcMaterializedSector]:
    runtime_schema = execution.get("runtime_schema")
    if not isinstance(runtime_schema, Mapping):
        raise ArtifactError("execution metadata has no runtime schema")
    amplitude_stage = runtime_schema.get("amplitude_stage")
    if not isinstance(amplitude_stage, Mapping):
        raise ArtifactError("runtime schema has no amplitude stage")
    roots = _mapping_records(amplitude_stage, "roots", "runtime amplitude roots")
    result: dict[int, _LcMaterializedSector] = {}
    for root in roots:
        sector_id = _strict_nonnegative_integer(
            root.get("color_sector_id"), "materialized root color sector ID"
        )
        group_id = _strict_nonnegative_integer(
            root.get("coherent_group_id"), "materialized coherent group ID"
        )
        reduction = reductions.get(group_id)
        if reduction is None:
            raise ArtifactError(
                "materialized LC sector is missing its physics reduction group"
            )
        computed = {
            color_index[identifier]
            for identifier in _string_members(
                reduction, "physical_color_ids", "reduction physical colors"
            )
            if colors[color_index[identifier]].get("computed") is True
        }
        if len(computed) != 1:
            raise ArtifactError(
                "materialized LC sector does not have one computed representative"
            )
        color = next(iter(computed))
        record = colors[color]
        if record.get("representative_id") != record.get("id"):
            raise ArtifactError(
                "computed LC materialized color does not represent itself"
            )
        weight = _decimal(
            root.get("all_sector_weight"), "materialized LC sector weight"
        )
        if weight <= 0:
            raise ArtifactError("materialized LC sector weight must be positive")
        sector = _LcMaterializedSector(color, weight)
        previous = result.setdefault(sector_id, sector)
        if previous != sector:
            raise ArtifactError(
                "materialized LC sector maps to inconsistent colors or weights"
            )
    return result


def _validate_lc_replay_proof(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ArtifactError("LC replay contract v2 group has no proof metadata")
    algorithm = value.get("algorithm")
    digest = value.get("digest")
    if (
        value.get("status") != "proven"
        or not isinstance(algorithm, str)
        or not algorithm
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArtifactError("LC replay contract v2 proof metadata is invalid")


def _lc_v2_label_mapping(
    value: object, external_count: int
) -> tuple[tuple[int, int], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError("LC replay label permutation is invalid")
    mapping = []
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            raise ArtifactError("LC replay label permutation entry is not an object")
        representative = _strict_nonnegative_integer(
            raw_item.get("representative_label"), "LC representative label"
        )
        sector = _strict_nonnegative_integer(
            raw_item.get("sector_label"), "LC sector label"
        )
        if (
            not 1 <= representative <= external_count
            or not 1 <= sector <= external_count
        ):
            raise ArtifactError("LC replay label permutation is out of range")
        if representative != sector:
            mapping.append((representative - 1, sector - 1))
    result = tuple(sorted(mapping))
    _validate_lc_public_mapping(result, external_count)
    return result


def _mapping_records(
    mapping: Mapping[str, object], key: str, context: str
) -> tuple[Mapping[str, object], ...]:
    value = mapping.get(key)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    if any(not isinstance(record, Mapping) for record in value):
        raise ArtifactError(f"{context} contain a non-object entry")
    return cast(tuple[Mapping[str, object], ...], tuple(value))


def _string_members(
    mapping: Mapping[str, object], key: str, context: str
) -> tuple[str, ...]:
    value = mapping.get(key)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ArtifactError(f"{context} contain duplicates")
    return result


def _nonnegative_integer_members(
    mapping: Mapping[str, object], key: str, context: str
) -> tuple[int, ...]:
    value = mapping.get(key)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    result = tuple(
        _strict_nonnegative_integer(item, f"{context} entry") for item in value
    )
    if len(set(result)) != len(result):
        raise ArtifactError(f"{context} contain duplicates")
    return result


def _validate_lc_public_mapping(
    mapping: Sequence[tuple[int, int]], external_count: int
) -> None:
    representatives: set[int] = set()
    sectors: set[int] = set()
    for representative, sector in mapping:
        if not 0 <= representative < external_count or not 0 <= sector < external_count:
            raise ArtifactError(
                "LC topology replay label permutation references an out-of-range leg"
            )
        if representative in representatives or sector in sectors:
            raise ArtifactError(
                "LC topology replay label permutation is not one-to-one"
            )
        representatives.add(representative)
        sectors.add(sector)
    if representatives != sectors:
        raise ArtifactError(
            "LC topology replay label permutation is not a permutation of its support"
        )


def _permute_lc_helicity(
    values: tuple[int, ...], mapping: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    target = list(values)
    for representative, sector in mapping:
        target[sector] = values[representative]
    return tuple(target)


def _permute_lc_color_word(
    word: tuple[int, ...], mapping: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    labels = {representative + 1: sector + 1 for representative, sector in mapping}
    return tuple(labels.get(label, label) for label in word)


def _lc_replay_public_words(
    word: tuple[int, ...], replay_weight: Decimal
) -> tuple[tuple[int, ...], ...]:
    tolerance = Decimal("1e-12")
    if abs(replay_weight - _ONE) <= tolerance:
        return (word,)
    if abs(replay_weight - _TWO) > tolerance:
        raise CompatibilityError(
            "resolved LC topology replay cannot expand sector weight "
            f"{replay_weight}; schema-v3 replay metadata defines public-flow "
            "expansion only for unit sectors and folded trace-reflection weight two"
        )
    if len(word) < 2:
        return (word,)
    reflected = (word[0], *reversed(word[1:]))
    return (word,) if reflected == word else (word, reflected)


def _apply_lc_replay_input_mapping(
    point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
    mapping: Sequence[tuple[int, int]],
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    mapped = list(point)
    for representative, sector in mapping:
        mapped[representative] = point[sector]
    return tuple(mapped)


def _apply_lc_replay_resolved(
    materialized: tuple[tuple[tuple[Decimal, ...], ...], ...],
    plan: _LcReplayPlan,
    point_count: int,
    helicity_ids: tuple[str, ...],
    color_ids: tuple[str, ...],
    selected_helicities: Sequence[str] | None,
    selected_colors: Sequence[str] | None,
) -> tuple[
    tuple[tuple[tuple[Decimal, ...], ...], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if len(helicity_ids) != plan.helicity_count or len(color_ids) != plan.color_count:
        raise ArtifactError(
            "materialized LC topology replay axes do not match public physics"
        )
    expected_points = point_count * len(plan.entries)
    if len(materialized) != expected_points:
        raise ArtifactError(
            "materialized LC topology replay result has an inconsistent point count"
        )
    full = [
        [[_ZERO for _ in range(plan.color_count)] for _ in range(plan.helicity_count)]
        for _ in range(point_count)
    ]
    for entry_index, entry in enumerate(plan.entries):
        for point_index in range(point_count):
            source = materialized[entry_index * point_count + point_index]
            if len(source) != plan.helicity_count or any(
                len(colors) != plan.color_count for colors in source
            ):
                raise ArtifactError(
                    "materialized LC topology replay result has an inconsistent shape"
                )
            for route in entry.routes:
                source_helicity, source_color = divmod(
                    route.source_index, plan.color_count
                )
                target_helicity, target_color = divmod(
                    route.target_index, plan.color_count
                )
                full[point_index][target_helicity][target_color] += (
                    route.weight * source[source_helicity][source_color]
                )
    selected_h = _selected_indices(helicity_ids, selected_helicities, "helicity")
    selected_c = _selected_indices(color_ids, selected_colors, "color component")
    values = tuple(
        tuple(tuple(full[p][h][c] for c in selected_c) for h in selected_h)
        for p in range(point_count)
    )
    return (
        values,
        tuple(helicity_ids[index] for index in selected_h),
        tuple(color_ids[index] for index in selected_c),
    )


def _evaluator_manifest(stage: object) -> Mapping[str, object]:
    if not isinstance(stage, Mapping) or not isinstance(
        stage.get("evaluator"), Mapping
    ):
        raise ArtifactError("serialized stage has no evaluator manifest")
    return cast(Mapping[str, object], stage["evaluator"])


def _runtime_state(native_runtime: Any) -> _RuntimeState:
    try:
        payload = json.loads(native_runtime._exact_runtime_state_json())
    except AttributeError as exc:
        raise CompatibilityError(
            "the installed Rusticol extension is too old for high-precision evaluation"
        ) from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError("Rusticol returned invalid exact-runtime state") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("Rusticol exact-runtime state is not an object")
    values = payload.get("model_parameter_values")
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ArtifactError("Rusticol exact-runtime model parameters are invalid")
    if "normalization_factor" not in payload:
        raise ArtifactError("Rusticol exact-runtime normalization is absent")
    return cast(_RuntimeState, payload)


def _prepare_points(
    momenta: Momenta,
    physics: Mapping[str, object],
    permutation: tuple[int, ...] | None,
) -> tuple[tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...], ...]:
    particles = physics.get("external_particles")
    if isinstance(particles, str | bytes) or not isinstance(particles, Sequence):
        raise ArtifactError("physics metadata has no external particles")
    external_count = len(particles)
    if not momenta:
        raise EvaluationError("momenta must contain at least one point")
    prepared = []
    for point_index, point in enumerate(momenta):
        if len(point) != external_count:
            raise EvaluationError(
                f"point {point_index} has {len(point)} legs, expected {external_count}"
            )
        legs = []
        for leg_index, leg in enumerate(point):
            if len(leg) != 4:
                raise EvaluationError(
                    f"point {point_index} leg {leg_index} must have four components"
                )
            legs.append(
                cast(
                    tuple[Decimal, Decimal, Decimal, Decimal],
                    tuple(
                        _decimal(
                            component,
                            f"point {point_index} leg {leg_index} "
                            f"component {component_index}",
                        )
                        for component_index, component in enumerate(leg)
                    ),
                )
            )
        if permutation is not None:
            legs = [legs[alias_index] for alias_index in permutation]
        prepared.append(tuple(legs))
    return tuple(prepared)


def _pack_stage_inputs(
    state: Sequence[_ComplexDecimal], stage: Mapping[str, object]
) -> tuple[_ComplexDecimal, ...]:
    parameter_count = _json_integer(stage["parameter_count"])
    raw_components = stage.get("input_components")
    if isinstance(raw_components, Sequence) and not isinstance(
        raw_components, str | bytes
    ):
        packed = [_complex_zero() for _ in range(parameter_count)]
        seen: set[int] = set()
        for raw_component in raw_components:
            component = cast(Mapping[str, object], raw_component)
            local = _json_integer(component["parameter_index"])
            global_index = _json_integer(component["global_component"])
            if local in seen or not 0 <= local < parameter_count:
                raise ArtifactError("stage input parameter mapping is invalid")
            try:
                packed[local] = state[global_index]
            except IndexError as exc:
                raise ArtifactError(
                    "stage input references an absent global value"
                ) from exc
            seen.add(local)
        if len(seen) != parameter_count:
            raise ArtifactError("stage input mapping is incomplete")
        return tuple(packed)
    if parameter_count != len(state):
        raise ArtifactError(
            "flat evaluator parameter count does not match runtime state"
        )
    return tuple(state)


def _assign_stage_outputs(
    state: list[_ComplexDecimal],
    outputs: Sequence[_ComplexDecimal],
    stage: Mapping[str, object],
) -> None:
    raw_slots = stage.get("output_slots")
    if isinstance(raw_slots, str | bytes) or not isinstance(raw_slots, Sequence):
        raise ArtifactError("stage output slots are invalid")
    for raw_slot in raw_slots:
        slot = cast(Mapping[str, object], raw_slot)
        output_start = _json_integer(slot["output_start"])
        output_stop = _json_integer(slot["output_stop"])
        component_start = _json_integer(slot["component_start"])
        component_stop = _json_integer(slot["component_stop"])
        if output_stop - output_start != component_stop - component_start:
            raise ArtifactError("stage output slot has inconsistent lengths")
        if output_stop > len(outputs) or component_stop > len(state):
            raise ArtifactError("stage output slot is out of range")
        state[component_start:component_stop] = outputs[output_start:output_stop]


def _fill_momenta(
    state: list[_ComplexDecimal],
    point: Sequence[tuple[Decimal, Decimal, Decimal, Decimal]],
    schema: Mapping[str, object],
) -> None:
    layout = cast(Mapping[str, object], schema["parameter_layout"])
    value_count = _json_integer(layout["value_component_count"])
    external = cast(Sequence[Mapping[str, object]], schema["external_particles"])
    incoming = {
        _json_integer(particle["label"])
        for particle in external
        if str(particle["role"]) == "initial"
    }
    for raw_slot in cast(Sequence[Mapping[str, object]], schema["momentum_slots"]):
        total = [_ZERO, _ZERO, _ZERO, _ZERO]
        for raw_label in cast(Sequence[object], raw_slot["external_labels"]):
            label = _json_integer(raw_label)
            sign = -_ONE if label in incoming else _ONE
            try:
                momentum = point[label - 1]
            except IndexError as exc:
                raise ArtifactError(
                    "momentum slot references an absent external leg"
                ) from exc
            for component in range(4):
                total[component] += sign * momentum[component]
        start = value_count + _json_integer(raw_slot["component_start"])
        for component, value in enumerate(total):
            state[start + component] = (value, _ZERO)


def _fill_sources(
    state: list[_ComplexDecimal],
    point: Sequence[tuple[Decimal, Decimal, Decimal, Decimal]],
    schema: Mapping[str, object],
    model_parameters: Sequence[Decimal],
) -> None:
    source_fill = cast(Mapping[str, object], schema["source_fill"])
    for raw_source in cast(Sequence[Mapping[str, object]], source_fill["sources"]):
        wave = _source_wavefunction(raw_source, point, schema, model_parameters)
        slot = cast(Mapping[str, object], raw_source["value_slot"])
        start = _json_integer(slot["component_start"])
        stop = _json_integer(slot["component_stop"])
        if stop - start != len(wave):
            raise ArtifactError("source wavefunction does not match its value slot")
        state[start:stop] = wave


def _fill_sources_with_states(
    state: list[_ComplexDecimal],
    point: Sequence[tuple[Decimal, Decimal, Decimal, Decimal]],
    schema: Mapping[str, object],
    model_parameters: Sequence[Decimal],
    source_states: Sequence[_ExactRuntimeSourceState | None],
) -> None:
    source_fill = cast(Mapping[str, object], schema["source_fill"])
    raw_sources = cast(Sequence[Mapping[str, object]], source_fill["sources"])
    if len(source_states) != len(raw_sources):
        raise ArtifactError(
            "helicity recurrence source-state count does not match runtime sources"
        )
    for raw_source, runtime_state in zip(raw_sources, source_states, strict=True):
        slot = cast(Mapping[str, object], raw_source["value_slot"])
        start = _json_integer(slot["component_start"])
        stop = _json_integer(slot["component_stop"])
        if not 0 <= start <= stop <= len(state):
            raise ArtifactError("source value slot is out of range")
        if runtime_state is None:
            state[start:stop] = [_complex_zero() for _ in range(stop - start)]
            continue
        source = dict(raw_source)
        source["source_helicity"] = runtime_state.helicity
        source["chirality"] = runtime_state.chirality
        source["spin_state"] = runtime_state.spin_state
        wave = tuple(
            _complex_mul(component, runtime_state.factor)
            for component in _source_wavefunction(
                source, point, schema, model_parameters
            )
        )
        if stop - start != len(wave):
            raise ArtifactError("source wavefunction does not match its value slot")
        state[start:stop] = wave


def _source_wavefunction(
    source: Mapping[str, object],
    point: Sequence[tuple[Decimal, Decimal, Decimal, Decimal]],
    schema: Mapping[str, object],
    model_parameters: Sequence[Decimal],
) -> tuple[_ComplexDecimal, ...]:
    leg_label = _json_integer(source["leg_label"])
    try:
        momentum = point[leg_label - 1]
    except IndexError as exc:
        raise ArtifactError("source references an absent external leg") from exc
    source_ir, identity, crossing = _validated_source_ir(source)
    if crossing["momentum_transform"] == "negate-four-momentum":
        momentum = cast(
            tuple[Decimal, Decimal, Decimal, Decimal],
            tuple(-component for component in momentum),
        )
    dimension = _json_integer(source_ir["component_dimension"])
    particle_id = _json_integer(identity["pdg_label"])
    anti_particle_id = _json_integer(identity["anti_pdg_label"])
    helicity = _json_integer(source["source_helicity"])
    chirality = _json_integer(source["chirality"])
    kind = str(source_ir["wavefunction_family"])
    orientation = str(identity["orientation"])
    wave: tuple[_ComplexDecimal, ...]
    if dimension == 1 and kind == "scalar":
        wave = ((_ONE, _ZERO),)
    elif kind == "fermion" and orientation == "self-conjugate":
        raise CompatibilityError(
            "self-conjugate fermion source wavefunctions are unsupported"
        )
    elif dimension == 2 and kind == "fermion":
        wave = (
            _antiquark_weyl(momentum, helicity, chirality)
            if orientation == "antiparticle"
            else _quark_weyl(momentum, helicity, chirality)
        )
    elif dimension == 4 and kind == "fermion":
        mass = _particle_mass(
            schema,
            particle_id,
            anti_particle_id,
            model_parameters,
        )
        wave = (
            _antiquark_dirac(momentum, helicity, mass)
            if orientation == "antiparticle"
            else _quark_dirac(momentum, helicity, mass)
        )
    elif dimension == 4 and kind == "vector":
        mass = _particle_mass(
            schema,
            particle_id,
            anti_particle_id,
            model_parameters,
        )
        wave = (
            _massless_vector(momentum, helicity)
            if mass == 0
            else _massive_vector(momentum, helicity, mass)
        )
    elif dimension == 16 and kind == "spin2":
        mass = _particle_mass(
            schema,
            particle_id,
            anti_particle_id,
            model_parameters,
        )
        wave = _spin2(momentum, helicity, mass)
    else:
        raise CompatibilityError(
            f"high-precision source kind {kind!r} with dimension {dimension} "
            "is unsupported"
        )
    phase = _crossing_phase(crossing)
    return tuple(_complex_mul(component, phase) for component in wave)


def _validated_source_ir(
    source: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    try:
        source_ir = cast(Mapping[str, object], source["source_ir"])
        identity = cast(Mapping[str, object], source_ir["identity"])
        declared_crossing = cast(Mapping[str, object], source_ir["crossing"])
        applied_crossing = cast(Mapping[str, object], source["applied_crossing"])
    except (KeyError, TypeError) as exc:
        raise ArtifactError(
            "source is missing typed SourceIR/CrossingIR metadata"
        ) from exc

    particle_id = _json_integer(identity["pdg_label"])
    anti_particle_id = _json_integer(identity["anti_pdg_label"])
    orientation = str(identity["orientation"])
    if orientation not in {"particle", "antiparticle", "self-conjugate"}:
        raise ArtifactError(f"source has invalid orientation {orientation!r}")
    if anti_particle_id == 0 or (
        (orientation == "self-conjugate") != (particle_id == anti_particle_id)
    ):
        raise ArtifactError(
            "source orientation is inconsistent with its antiparticle relation"
        )

    side = str(source["side"])
    expected_crossing = (
        declared_crossing
        if side == "initial"
        else {
            "momentum_transform": "identity",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0],
        }
    )
    if dict(applied_crossing) != dict(expected_crossing):
        raise ArtifactError(
            "source applied crossing is inconsistent with its side and SourceIR"
        )
    transform = str(applied_crossing["momentum_transform"])
    if transform not in {"identity", "negate-four-momentum"}:
        raise ArtifactError(f"source has unsupported momentum transform {transform!r}")
    _crossing_phase(applied_crossing)

    dimension = _json_integer(source_ir["component_dimension"])
    family = str(source_ir["wavefunction_family"])
    basis = str(source_ir["basis"])
    flattened = {
        "particle_id": particle_id,
        "anti_particle_id": anti_particle_id,
        "source_orientation": orientation,
        "wavefunction_kind": family,
        "dimension": dimension,
        "source_basis": basis,
        "crossing": (
            "negate-incoming-momentum"
            if transform == "negate-four-momentum"
            else "identity"
        ),
    }
    for key, expected in flattened.items():
        actual = source.get(key)
        if actual != expected:
            raise ArtifactError(
                f"source flattened field {key!r} disagrees with typed metadata"
            )

    state = (
        _json_integer(source["source_helicity"]),
        _json_integer(source["chirality"]),
        _hashable_spin_state(source["spin_state"]),
    )
    transformed_states = {
        _apply_source_crossing_state(
            declared,
            applied_crossing,
        )
        for declared in cast(Sequence[Mapping[str, object]], source_ir["states"])
    }
    if state not in transformed_states:
        raise ArtifactError("source state is not declared by its typed SourceIR")
    return source_ir, identity, applied_crossing


def _apply_source_crossing_state(
    state: Mapping[str, object],
    crossing: Mapping[str, object],
) -> tuple[int, int, object]:
    spin_state = state["spin_state"]
    spin_factor = _json_integer(crossing["spin_state_factor"])
    if spin_factor != 1:
        if isinstance(spin_state, bool) or not isinstance(spin_state, int):
            raise ArtifactError(
                "crossing cannot multiply a structured source spin state"
            )
        spin_state *= spin_factor
    return (
        _json_integer(state["helicity"]) * _json_integer(crossing["helicity_factor"]),
        _json_integer(state["chirality"]) * _json_integer(crossing["chirality_factor"]),
        _hashable_spin_state(spin_state),
    )


def _crossing_phase(crossing: Mapping[str, object]) -> _ComplexDecimal:
    raw_phase = crossing.get("phase")
    if not isinstance(raw_phase, Sequence) or isinstance(raw_phase, (str, bytes)):
        raise ArtifactError("source crossing phase must be a complex pair")
    if len(raw_phase) != 2:
        raise ArtifactError("source crossing phase must be a complex pair")
    try:
        phase = (Decimal(str(raw_phase[0])), Decimal(str(raw_phase[1])))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ArtifactError(
            "source crossing phase must be a finite complex pair"
        ) from exc
    if not phase[0].is_finite() or not phase[1].is_finite():
        raise ArtifactError("source crossing phase must be a finite complex pair")
    if phase == (_ZERO, _ZERO):
        raise ArtifactError("source crossing phase must be nonzero")
    return phase


def _hashable_spin_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_hashable_spin_state(item) for item in value)
    return value


def _particle_mass(
    schema: Mapping[str, object],
    particle_id: int,
    anti_particle_id: int,
    parameters: Sequence[Decimal],
) -> Decimal:
    model = cast(Mapping[str, object], schema["model"])
    particles = cast(Sequence[Mapping[str, object]], model["particles"])
    record = next(
        (item for item in particles if _json_integer(item["pdg"]) == particle_id),
        None,
    )
    if record is None:
        record = next(
            (
                item
                for item in particles
                if _json_integer(item["pdg"]) == anti_particle_id
            ),
            None,
        )
    if record is None:
        return _ZERO
    parameter_name = record.get("mass_parameter")
    if isinstance(parameter_name, str):
        for raw_parameter in cast(
            Sequence[Mapping[str, object]], schema["model_parameters"]
        ):
            if raw_parameter["name"] == parameter_name:
                try:
                    return parameters[_json_integer(raw_parameter["parameter_index"])]
                except IndexError as exc:
                    raise ArtifactError(
                        "particle mass parameter index is invalid"
                    ) from exc
    return _decimal(record.get("mass", 0), "particle mass")


def _quark_weyl(
    momentum: Sequence[Decimal], helicity: int, chirality: int
) -> tuple[_ComplexDecimal, _ComplexDecimal]:
    energy, px, py, pz = momentum
    if energy > 0:
        sq = (
            _ZERO
            if px == 0 and py == 0 and pz < 0
            else _sqrt(energy + pz, "Weyl source")
        )
        chi1 = (sq, _ZERO)
        chi2 = (
            (-Decimal(helicity) * _sqrt(_TWO * energy, "Weyl source"), _ZERO)
            if sq == 0
            else (Decimal(helicity) * px / sq, -py / sq)
        )
        if helicity == 1 and chirality == 1:
            return chi1, chi2
        if helicity == -1 and chirality == -1:
            return chi2, chi1
        return _complex_zero(), _complex_zero()
    sq = (
        _ZERO
        if px == 0 and py == 0 and pz > 0
        else -_sqrt(-(energy + pz), "Weyl source")
    )
    chi1 = (sq, _ZERO)
    chi2 = (
        (-Decimal(helicity) * _sqrt(_TWO * abs(energy), "Weyl source"), _ZERO)
        if sq == 0
        else (Decimal(helicity) * px / sq, py / sq)
    )
    if helicity == -1 and chirality == 1:
        return chi1, chi2
    if helicity == 1 and chirality == -1:
        return chi2, chi1
    return _complex_zero(), _complex_zero()


def _antiquark_weyl(
    momentum: Sequence[Decimal], helicity: int, chirality: int
) -> tuple[_ComplexDecimal, _ComplexDecimal]:
    energy, px, py, pz = momentum
    if energy > 0:
        sq = (
            _ZERO
            if px == 0 and py == 0 and pz < 0
            else -_sqrt(energy + pz, "Weyl source")
        )
        chi1 = (sq, _ZERO)
        chi2 = (
            (-Decimal(helicity) * _sqrt(_TWO * energy, "Weyl source"), _ZERO)
            if sq == 0
            else (-Decimal(helicity) * px / sq, py / sq)
        )
        if helicity == 1 and chirality == 1:
            return chi2, chi1
        if helicity == -1 and chirality == -1:
            return chi1, chi2
        return _complex_zero(), _complex_zero()
    sq = (
        _ZERO
        if px == 0 and py == 0 and pz > 0
        else _sqrt(-(energy + pz), "Weyl source")
    )
    chi1 = (sq, _ZERO)
    chi2 = (
        (-Decimal(helicity) * _sqrt(_TWO * abs(energy), "Weyl source"), _ZERO)
        if sq == 0
        else (-Decimal(helicity) * px / sq, -py / sq)
    )
    if helicity == -1 and chirality == 1:
        return chi2, chi1
    if helicity == 1 and chirality == -1:
        return chi1, chi2
    return _complex_zero(), _complex_zero()


def _massless_quark_dirac(
    momentum: Sequence[Decimal], helicity: int
) -> tuple[_ComplexDecimal, ...]:
    left, right = _quark_weyl(momentum, helicity, helicity)
    return (
        (left, right, _complex_zero(), _complex_zero())
        if helicity == 1
        else (_complex_zero(), _complex_zero(), left, right)
    )


def _massless_antiquark_dirac(
    momentum: Sequence[Decimal], helicity: int
) -> tuple[_ComplexDecimal, ...]:
    first, second = _antiquark_weyl(momentum, helicity, helicity)
    return (
        (_complex_zero(), _complex_zero(), first, second)
        if -helicity == 1
        else (first, second, _complex_zero(), _complex_zero())
    )


def _quark_dirac(
    momentum: Sequence[Decimal], helicity: int, mass: Decimal
) -> tuple[_ComplexDecimal, ...]:
    if mass == 0:
        return _massless_quark_dirac(momentum, helicity)
    return _massive_dirac(momentum, helicity, mass, antiquark=False)


def _antiquark_dirac(
    momentum: Sequence[Decimal], helicity: int, mass: Decimal
) -> tuple[_ComplexDecimal, ...]:
    if mass == 0:
        return _massless_antiquark_dirac(momentum, helicity)
    return _massive_dirac(momentum, helicity, mass, antiquark=True)


def _massive_dirac(
    momentum: Sequence[Decimal], helicity: int, mass: Decimal, *, antiquark: bool
) -> tuple[_ComplexDecimal, ...]:
    energy, px, py, pz = momentum
    nsf = (-1 if energy > 0 else 1) if antiquark else (1 if energy > 0 else -1)
    nh = nsf * helicity
    pp = _sqrt(px * px + py * py + pz * pz, "massive fermion source")
    omega1 = _sqrt(abs(energy) + pp, "massive fermion source")
    if omega1 == 0:
        raise EvaluationError("massive fermion source has zero normalization")
    omega = (omega1, mass / omega1)
    half = _ONE / _TWO
    sf1 = Decimal(1 + nsf + (1 - nsf) * nh) * half
    sf2 = Decimal(1 + nsf - (1 - nsf) * nh) * half
    ip = (3 + nh) // 2 - 1
    im = (3 - nh) // 2 - 1
    sfomeg = (sf1 * omega[ip], sf2 * omega[im])
    signed_px, signed_py, signed_pz = (px, py, pz) if energy > 0 else (-px, -py, -pz)
    pp3 = max(pp + signed_pz, _ZERO)
    chi1 = _ONE if pp == 0 else _sqrt(pp3 / (_TWO * pp), "massive fermion source")
    if pp3 == 0 or pp == 0:
        chi2 = (-Decimal(nh), _ZERO)
    else:
        denominator = _sqrt(_TWO * pp * pp3, "massive fermion source")
        chi2 = (
            Decimal(nh) * signed_px / denominator,
            (signed_py if antiquark else -signed_py) / denominator,
        )
    chi = ((chi1, _ZERO), chi2)
    if antiquark:
        return (
            (chi[im][0] * sfomeg[0], chi[im][1] * sfomeg[0]),
            (chi[ip][0] * sfomeg[0], chi[ip][1] * sfomeg[0]),
            (chi[im][0] * sfomeg[1], chi[im][1] * sfomeg[1]),
            (chi[ip][0] * sfomeg[1], chi[ip][1] * sfomeg[1]),
        )
    return (
        (chi[im][0] * sfomeg[1], chi[im][1] * sfomeg[1]),
        (chi[ip][0] * sfomeg[1], chi[ip][1] * sfomeg[1]),
        (chi[im][0] * sfomeg[0], chi[im][1] * sfomeg[0]),
        (chi[ip][0] * sfomeg[0], chi[ip][1] * sfomeg[0]),
    )


def _massless_vector(
    momentum: Sequence[Decimal], helicity: int
) -> tuple[_ComplexDecimal, ...]:
    energy, px, py, pz = momentum
    if energy == 0:
        raise EvaluationError("cannot build a massless vector with zero energy")
    sqh = _sqrt(_ONE / _TWO, "vector source")
    if energy > 0:
        hel = Decimal(helicity)
        pp = energy
        pt = _sqrt(px * px + py * py, "vector source")
        wf3 = (hel * pt / pp * sqh, _ZERO)
        if pt != 0:
            pzpt = pz / (pp * pt) * sqh * hel
            wf1 = (-px * pzpt, -py / pt * sqh)
            wf2 = (-py * pzpt, px / pt * sqh)
        else:
            wf1 = (-hel * sqh, _ZERO)
            wf2 = (_ZERO, _fortran_sign(sqh, pz))
        return _complex_zero(), wf1, wf2, wf3
    hel = Decimal(-helicity)
    pp = -energy
    pt = _sqrt(px * px + py * py, "vector source")
    wf3 = (hel * pt / pp * sqh, _ZERO)
    if pt != 0:
        pzpt = -pz / (pp * pt) * sqh * hel
        wf1 = (px * pzpt, py / pt * sqh)
        wf2 = (py * pzpt, -px / pt * sqh)
    else:
        wf1 = (-hel * sqh, _ZERO)
        wf2 = (_ZERO, -_fortran_sign(sqh, pz))
    return _complex_zero(), wf1, wf2, wf3


def _massive_vector(
    momentum: Sequence[Decimal], helicity: int, mass: Decimal
) -> tuple[_ComplexDecimal, ...]:
    energy, px, py, pz = momentum
    if mass == 0:
        raise EvaluationError("massive-vector source has zero mass")
    if energy < 0:
        return _massive_vector(
            tuple(-component for component in momentum),
            -helicity,
            mass,
        )
    sqh = _sqrt(_ONE / _TWO, "massive vector source")
    hel = Decimal(helicity)
    nsvahl = Decimal(abs(helicity))
    pt2 = px * px + py * py
    pp = min(energy, _sqrt(pt2 + pz * pz, "massive vector source"))
    pt = min(pp, _sqrt(pt2, "massive vector source"))
    hel0 = _ONE - abs(hel)
    if pp == 0:
        return (
            _complex_zero(),
            (-hel * sqh, _ZERO),
            (_ZERO, nsvahl * sqh),
            (hel0, _ZERO),
        )
    emp = energy / (mass * pp)
    wf0 = (hel0 * pp / mass, _ZERO)
    wf3 = (hel0 * pz * emp + hel * pt / pp * sqh, _ZERO)
    if pt != 0:
        pzpt = pz / (pp * pt) * sqh * hel
        wf1 = (hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh)
        wf2 = (hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh)
    else:
        wf1 = (-hel * sqh, _ZERO)
        wf2 = (_ZERO, nsvahl * _fortran_sign(sqh, pz))
    return wf0, wf1, wf2, wf3


def _spin2(
    momentum: Sequence[Decimal], helicity: int, mass: Decimal
) -> tuple[_ComplexDecimal, ...]:
    if mass == 0:
        if helicity not in {-2, 2}:
            raise EvaluationError("massless spin-2 source supports helicity +/-2")
        vector = _massless_vector(momentum, helicity // 2)
        return _spin2_outer(vector, vector)
    plus = _massive_vector(momentum, 1, mass)
    minus = _massive_vector(momentum, -1, mass)
    longitudinal = _massive_vector(momentum, 0, mass)
    if helicity == 2:
        return _spin2_outer(plus, plus)
    if helicity == -2:
        return _spin2_outer(minus, minus)
    if helicity == 1:
        return _spin2_sum(
            (_spin2_outer(plus, longitudinal), _ONE / _sqrt(_TWO, "spin-2 source")),
            (_spin2_outer(longitudinal, plus), _ONE / _sqrt(_TWO, "spin-2 source")),
        )
    if helicity == -1:
        return _spin2_sum(
            (_spin2_outer(minus, longitudinal), _ONE / _sqrt(_TWO, "spin-2 source")),
            (_spin2_outer(longitudinal, minus), _ONE / _sqrt(_TWO, "spin-2 source")),
        )
    if helicity != 0:
        raise EvaluationError(f"unsupported massive spin-2 helicity {helicity}")
    sqrt6 = _sqrt(Decimal(6), "spin-2 source")
    return _spin2_sum(
        (_spin2_outer(plus, minus), _ONE / sqrt6),
        (_spin2_outer(minus, plus), _ONE / sqrt6),
        (_spin2_outer(longitudinal, longitudinal), _TWO / sqrt6),
    )


def _spin2_outer(
    left: Sequence[_ComplexDecimal], right: Sequence[_ComplexDecimal]
) -> tuple[_ComplexDecimal, ...]:
    return tuple(
        _complex_mul(left[mu], right[nu]) for mu in range(4) for nu in range(4)
    )


def _spin2_sum(
    *terms: tuple[Sequence[_ComplexDecimal], Decimal],
) -> tuple[_ComplexDecimal, ...]:
    values = []
    for index in range(16):
        real = sum((weight * tensor[index][0] for tensor, weight in terms), _ZERO)
        imaginary = sum((weight * tensor[index][1] for tensor, weight in terms), _ZERO)
        values.append((real, imaginary))
    return tuple(values)


def _reduce_materialized_helicity(
    amplitudes: Sequence[_ComplexDecimal],
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    normalization: Decimal,
    helicity_index: int,
    root_factors: Sequence[_ComplexDecimal | None],
) -> tuple[Decimal, ...]:
    helicities = cast(Sequence[Mapping[str, object]], physics["helicities"])
    colors = cast(Sequence[Mapping[str, object]], physics["color_components"])
    try:
        helicity_id = str(helicities[helicity_index]["id"])
    except IndexError as exc:
        raise ArtifactError(
            "helicity recurrence selected an absent physical helicity"
        ) from exc
    runtime_schema = execution.get("runtime_schema")
    if not isinstance(runtime_schema, Mapping):
        raise ArtifactError("execution metadata has no runtime schema")
    amplitude_stage = runtime_schema.get("amplitude_stage")
    if not isinstance(amplitude_stage, Mapping):
        raise ArtifactError("runtime schema has no amplitude stage")
    raw_roots = amplitude_stage.get("roots")
    if isinstance(raw_roots, str | bytes) or not isinstance(raw_roots, Sequence):
        raise ArtifactError("amplitude roots are invalid")
    roots = cast(Sequence[Mapping[str, object]], raw_roots)
    if len(amplitudes) != len(roots) or len(root_factors) != len(roots):
        raise ArtifactError(
            "helicity recurrence amplitude output width is inconsistent"
        )

    groups: dict[int, list[int]] = {}
    all_sector_weights: dict[int, Decimal] = {}
    for index, root in enumerate(roots):
        output_index = _strict_nonnegative_integer(
            root.get("output_index"), "amplitude root output index"
        )
        if output_index != index:
            raise ArtifactError("amplitude root outputs are not contiguous")
        group_id = _json_integer(root.get("coherent_group_id"))
        groups.setdefault(group_id, []).append(output_index)
        weight = _decimal(
            root.get("all_sector_weight", root.get("helicity_weight")),
            "all-sector weight",
        )
        previous = all_sector_weights.setdefault(group_id, weight)
        if previous != weight:
            raise ArtifactError("coherent amplitude group has inconsistent weights")

    reduction_metadata = physics.get("reduction")
    if not isinstance(reduction_metadata, Mapping):
        raise ArtifactError("resolved physics has no reduction metadata")
    raw_reductions = reduction_metadata.get("groups")
    if isinstance(raw_reductions, str | bytes) or not isinstance(
        raw_reductions, Sequence
    ):
        raise ArtifactError("resolved physics reduction groups are invalid")
    reductions: dict[int, Mapping[str, object]] = {}
    for raw_reduction in raw_reductions:
        if not isinstance(raw_reduction, Mapping):
            raise ArtifactError("resolved physics reduction group is invalid")
        identifier = str(raw_reduction.get("id", ""))
        if not identifier.startswith("reduction:"):
            raise ArtifactError("resolved reduction group has an invalid ID")
        try:
            group_id = int(identifier.removeprefix("reduction:"))
        except ValueError as exc:
            raise ArtifactError("resolved reduction group has an invalid ID") from exc
        if group_id in reductions:
            raise ArtifactError("resolved physics repeats a reduction group")
        reductions[group_id] = raw_reduction
    if set(groups) - set(reductions):
        raise ArtifactError("resolved physics is missing an amplitude reduction group")

    coherent: dict[int, _ComplexDecimal] = {}
    active_groups: set[int] = set()
    for group_id, indices in groups.items():
        reduction = reductions[group_id]
        members = tuple(
            str(value)
            for value in _mapping_sequence(
                reduction,
                "physical_helicity_ids",
                "reduction physical helicities",
            )
        )
        if helicity_id not in members:
            continue
        values = [
            _complex_mul(amplitudes[index], factor)
            for index in indices
            if (factor := root_factors[index]) is not None
        ]
        if not values:
            continue
        coherent[group_id] = (
            sum((value[0] for value in values), _ZERO),
            sum((value[1] for value in values), _ZERO),
        )
        active_groups.add(group_id)

    contraction = amplitude_stage.get("color_contraction")
    if isinstance(contraction, Mapping):
        if (
            len(colors) != 1
            or str(physics.get("color_accuracy")) == "lc"
            or colors[0].get("kind") == "lc-flow"
        ):
            raise ArtifactError("contracted color reduction requires one color axis")
        entries = _mapping_sequence(contraction, "entries", "color contraction entries")
        total = _ZERO
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                raise ArtifactError("color contraction entry is not an object")
            left_id = _json_integer(raw_entry.get("left_group_id"))
            right_id = _json_integer(raw_entry.get("right_group_id"))
            if left_id not in groups or right_id not in groups:
                raise ArtifactError(
                    "color contraction references an unknown coherent group"
                )
            if left_id not in active_groups or right_id not in active_groups:
                continue
            left = coherent[left_id]
            right = coherent[right_id]
            product_re = left[0] * right[0] + left[1] * right[1]
            product_im = left[1] * right[0] - left[0] * right[1]
            contraction_weight = _complex_decimal_pair(
                raw_entry.get("weight"), "color contraction weight"
            )
            total += (
                normalization
                * _decimal(raw_entry.get("symmetry_factor", 1), "color symmetry factor")
                * (
                    contraction_weight[0] * product_re
                    - contraction_weight[1] * product_im
                )
            )
        return (total,)
    if contraction is not None:
        raise ArtifactError("color contraction metadata is not an object")

    color_index = {str(color["id"]): index for index, color in enumerate(colors)}
    if len(color_index) != len(colors):
        raise ArtifactError("resolved physics contains duplicate color IDs")
    full = [_ZERO for _ in colors]
    for group_id in active_groups:
        value = coherent[group_id]
        reduction = reductions[group_id]
        member_ids = tuple(
            str(value)
            for value in _mapping_sequence(
                reduction, "physical_color_ids", "reduction physical colors"
            )
        )
        member_weights: list[tuple[int, Decimal]] = []
        for identifier in member_ids:
            color_slot = color_index.get(identifier)
            if color_slot is None:
                raise ArtifactError(
                    f"reduction references unknown color component {identifier!r}"
                )
            weight = _decimal(
                colors[color_slot].get("coefficient", 1), "color coefficient"
            )
            if weight <= 0:
                raise ArtifactError("reduction color coefficient must be positive")
            member_weights.append((color_slot, weight))
        total_weight = sum((weight for _index, weight in member_weights), _ZERO)
        if total_weight <= 0:
            raise ArtifactError("reduction group has no positive color weight")
        contribution = (
            normalization
            * all_sector_weights[group_id]
            * (value[0] * value[0] + value[1] * value[1])
        )
        for index, weight in member_weights:
            full[index] += contribution * weight / total_weight
    return tuple(full)


def _mapping_sequence(
    mapping: Mapping[str, object], key: str, context: str
) -> Sequence[object]:
    value = mapping.get(key)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    return value


def _reduce_resolved(
    amplitudes: Sequence[Sequence[_ComplexDecimal]],
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    normalization: Decimal,
    selected_helicities: Sequence[str] | None,
    selected_colors: Sequence[str] | None,
) -> tuple[
    tuple[tuple[tuple[Decimal, ...], ...], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    helicities = cast(Sequence[Mapping[str, object]], physics["helicities"])
    colors = cast(Sequence[Mapping[str, object]], physics["color_components"])
    helicity_ids = tuple(str(item["id"]) for item in helicities)
    color_ids = tuple(str(item["id"]) for item in colors)
    selected_h = _selected_indices(helicity_ids, selected_helicities, "helicity")
    selected_c = _selected_indices(color_ids, selected_colors, "color component")
    accuracy = str(physics["color_accuracy"])
    if accuracy != "lc" and selected_colors is not None:
        raise EvaluationError(
            "LC color-flow selection is unavailable for NLC/full artifacts"
        )

    amplitude_stage = cast(
        Mapping[str, object],
        cast(Mapping[str, object], execution["runtime_schema"])["amplitude_stage"],
    )
    roots = cast(Sequence[Mapping[str, object]], amplitude_stage["roots"])
    groups: dict[int, list[int]] = {}
    all_sector_weights: dict[int, Decimal] = {}
    for root in roots:
        group_id = _json_integer(root["coherent_group_id"])
        groups.setdefault(group_id, []).append(_json_integer(root["output_index"]))
        weight = root.get("all_sector_weight", root["helicity_weight"])
        all_sector_weights[group_id] = _decimal(weight, "all-sector weight")

    reductions = {
        int(str(group["id"]).removeprefix("reduction:")): group
        for group in cast(
            Sequence[Mapping[str, object]],
            cast(Mapping[str, object], physics["reduction"])["groups"],
        )
    }
    helicity_index = {
        identifier: index for index, identifier in enumerate(helicity_ids)
    }
    color_index = {identifier: index for index, identifier in enumerate(color_ids)}
    full_points = []
    contraction = amplitude_stage.get("color_contraction")
    for raw_amplitudes in amplitudes:
        coherent = {
            group_id: (
                sum((raw_amplitudes[index][0] for index in indices), _ZERO),
                sum((raw_amplitudes[index][1] for index in indices), _ZERO),
            )
            for group_id, indices in groups.items()
        }
        full = [[_ZERO for _ in color_ids] for _ in helicity_ids]
        if isinstance(contraction, Mapping):
            if len(color_ids) != 1:
                raise ArtifactError(
                    "contracted color reduction requires one color axis"
                )
            for entry in cast(Sequence[Mapping[str, object]], contraction["entries"]):
                left_id = _json_integer(entry["left_group_id"])
                right_id = _json_integer(entry["right_group_id"])
                left = coherent[left_id]
                right = coherent[right_id]
                product_re = left[0] * right[0] + left[1] * right[1]
                product_im = left[1] * right[0] - left[0] * right[1]
                weight = cast(Sequence[object], entry["weight"])
                contribution = (
                    normalization
                    * _decimal(entry.get("symmetry_factor", 1), "color symmetry factor")
                    * (
                        _decimal(weight[0], "color weight") * product_re
                        - _decimal(weight[1], "color weight") * product_im
                    )
                )
                reduction = reductions[left_id]
                if (
                    reduction["physical_helicity_ids"]
                    != reductions[right_id]["physical_helicity_ids"]
                ):
                    raise ArtifactError("color contraction mixes physical helicities")
                contracted_members = tuple(
                    str(value)
                    for value in cast(
                        Sequence[object], reduction["physical_helicity_ids"]
                    )
                )
                member_weights = [
                    _decimal(
                        helicities[helicity_index[item]]["coefficient"],
                        "helicity coefficient",
                    )
                    for item in contracted_members
                ]
                total_weight = sum(member_weights, _ZERO)
                if total_weight <= 0:
                    raise ArtifactError(
                        "reduction group has no positive helicity weight"
                    )
                for identifier, member_weight in zip(
                    contracted_members, member_weights, strict=True
                ):
                    full[helicity_index[identifier]][0] += (
                        contribution * member_weight / total_weight
                    )
        else:
            for group_id, value in coherent.items():
                reduction = reductions[group_id]
                contribution = (
                    normalization
                    * all_sector_weights[group_id]
                    * (value[0] * value[0] + value[1] * value[1])
                )
                diagonal_members: list[tuple[int, int, Decimal]] = []
                total_weight = _ZERO
                for helicity_id in cast(
                    Sequence[object], reduction["physical_helicity_ids"]
                ):
                    h_id = str(helicity_id)
                    h_weight = _decimal(
                        helicities[helicity_index[h_id]]["coefficient"],
                        "helicity coefficient",
                    )
                    for color_id in cast(
                        Sequence[object], reduction["physical_color_ids"]
                    ):
                        c_id = str(color_id)
                        color_record = colors[color_index[c_id]]
                        c_weight = _decimal(
                            color_record.get("coefficient", 1), "color coefficient"
                        )
                        weight = h_weight * c_weight
                        diagonal_members.append(
                            (helicity_index[h_id], color_index[c_id], weight)
                        )
                        total_weight += weight
                if total_weight <= 0:
                    raise ArtifactError("reduction group has no positive member weight")
                for h_index, c_index, weight in diagonal_members:
                    full[h_index][c_index] += contribution * weight / total_weight
        full_points.append(
            tuple(tuple(full[h][c] for c in selected_c) for h in selected_h)
        )
    return (
        tuple(full_points),
        tuple(helicity_ids[index] for index in selected_h),
        tuple(color_ids[index] for index in selected_c),
    )


def _selected_indices(
    available: Sequence[str], requested: Sequence[str] | None, kind: str
) -> tuple[int, ...]:
    if requested is None:
        return tuple(range(len(available)))
    if not requested:
        raise EvaluationError(f"{kind} selection must not be empty")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise EvaluationError(f"unknown resolved {kind} ID {unknown[0]!r}")
    requested_set = set(requested)
    return tuple(
        index
        for index, identifier in enumerate(available)
        if identifier in requested_set
    )


__all__ = ["SymbolicaExactExecutor"]
