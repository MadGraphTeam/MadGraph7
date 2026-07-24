# SPDX-License-Identifier: 0BSD
"""Public exact executor for compact recurrence artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.api.results import ResolvedEvaluation
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import confined_path
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.eager_exact._contracts import (
    _KernelLoader,
    _mapping,
    _PayloadIndex,
    _read_json,
    _selected_process,
    _sequence,
)
from pyamplicol.runtime.symbolica_exact import (
    _decimal,
    _prepare_points,
    _runtime_state,
    _selected_indices,
    _working_precision,
)

from ._execution import _evaluate_replay_point, _evaluate_union_point
from ._plan import _RecurrenceExactPlan
from ._plan_v2 import (
    DIRECT_NONE_U32,
    _AmplitudeDestination,
    _NativeExactSectionsLoader,
    _ReplayTarget,
    _ResolvedHelicity,
)

_ZERO = Decimal(0)


class RecurrenceExactExecutor:
    """Execute one authenticated compact recurrence through exact kernels."""

    def __init__(
        self,
        artifact: Path,
        process_id: str,
        native_runtime: Any,
        *,
        kernel_loader: _KernelLoader | None = None,
        native_sections_loader: _NativeExactSectionsLoader | None = None,
    ) -> None:
        self._artifact = Path(artifact).expanduser().resolve(strict=True)
        self._native_runtime = native_runtime
        manifest = load_manifest(self._artifact)
        exact_payloads = ExactEvaluatorPayloadResolver(manifest)
        process, permutation = _selected_process(manifest.processes, process_id)
        representative_id = str(process["id"])
        execution_records = tuple(
            record
            for record in manifest.payloads
            if record.role == "evaluator-manifest"
            and record.process_id == representative_id
        )
        if len(execution_records) != 1:
            raise ArtifactError(
                f"process {representative_id!r} must declare one recurrence "
                "execution manifest"
            )
        physics_path = process.get("physics_path")
        if not isinstance(physics_path, str):
            raise ArtifactError(f"process {representative_id!r} has no physics path")
        payloads = _PayloadIndex.from_manifest(manifest)
        payloads.require(
            physics_path,
            role="runtime-physics",
            process_id=representative_id,
        )
        execution = _read_json(
            confined_path(self._artifact, execution_records[0].path),
            "recurrence execution metadata",
        )
        physics = _read_json(
            confined_path(self._artifact, physics_path),
            "runtime physics metadata",
        )
        self._plan = _RecurrenceExactPlan.load(
            artifact_root=self._artifact,
            process_id=representative_id,
            execution=execution,
            manifest=manifest,
            kernel_loader=kernel_loader,
            exact_payloads=exact_payloads,
            native_sections_loader=native_sections_loader,
        )
        self._physics = physics
        self._permutation = permutation
        if self._plan.sections.strategy == "topology-replay":
            self._replay_by_color = self._replay_targets_by_color()
            self._destination_helicities = self._destination_helicity_maps()
            self._union_destination_by_color = ()
            self._union_helicity_by_physics = ()
        else:
            self._replay_by_color = ()
            self._destination_helicities = ()
            self._union_destination_by_color = self._union_destinations_by_color()
            self._union_helicity_by_physics = self._union_helicities_by_physics()

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        precision: int,
    ) -> ResolvedEvaluation:
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise EvaluationError(
                "precision must be a positive integer number of decimal digits"
            )
        working_precision = _working_precision(precision)
        points = _prepare_points(momenta, self._physics, self._permutation)
        state = _runtime_state(self._native_runtime)
        runtime_parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state["model_parameter_values"]
        )
        normalization = _decimal(
            state["normalization_factor"],
            "runtime normalization",
        )
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        color_records = tuple(
            _mapping(value, f"physics color {index}")
            for index, value in enumerate(
                _sequence(
                    self._physics.get("color_components"),
                    "physics color components",
                )
            )
        )
        helicity_ids = tuple(str(record["id"]) for record in helicity_records)
        color_ids = tuple(str(record["id"]) for record in color_records)
        selected_helicities = _selected_indices(
            helicity_ids,
            helicities,
            "helicity",
        )
        selected_colors = _selected_indices(
            color_ids,
            color_flows,
            "color component",
        )
        helicity_positions = {
            index: position for position, index in enumerate(selected_helicities)
        }

        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            parameters = self._plan.resolve_model_parameters(
                runtime_parameters,
                working_precision,
            )
            values = []
            for point in points:
                point_values = [
                    [_ZERO for _ in selected_colors] for _ in selected_helicities
                ]
                if self._plan.sections.strategy == "topology-replay":
                    self._evaluate_replay_resolved_point(
                        point,
                        selected_colors,
                        helicity_positions,
                        helicity_records,
                        color_records,
                        parameters.prepared,
                        working_precision,
                        normalization,
                        point_values,
                    )
                else:
                    self._evaluate_union_resolved_point(
                        point,
                        selected_helicities,
                        selected_colors,
                        helicity_records,
                        color_records,
                        parameters.prepared,
                        working_precision,
                        normalization,
                        point_values,
                    )
                values.append(tuple(tuple(colors) for colors in point_values))

        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            rounded = tuple(
                tuple(tuple(+entry for entry in colors) for colors in point)
                for point in values
            )
        return ResolvedEvaluation(
            values=rounded,
            helicity_ids=tuple(helicity_ids[index] for index in selected_helicities),
            color_ids=tuple(color_ids[index] for index in selected_colors),
            color_accuracy=cast(Any, str(self._physics["color_accuracy"])),
        )

    def _evaluate_replay_resolved_point(
        self,
        point: object,
        selected_colors: Sequence[int],
        helicity_positions: dict[int, int],
        helicity_records: Sequence[dict[str, object]],
        color_records: Sequence[dict[str, object]],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        working_precision: int,
        normalization: Decimal,
        point_values: list[list[Decimal]],
    ) -> None:
        for color_position, color_index in enumerate(selected_colors):
            target = self._replay_by_color[color_index]
            amplitudes = _evaluate_replay_point(
                self._plan,
                cast(Any, point),
                target,
                prepared_parameters,
                working_precision,
            )
            destination_helicities = self._destination_helicities[color_index]
            color_weight = _decimal(
                color_records[color_index].get("coefficient", 1),
                "color coefficient",
            )
            for destination in self._plan.sections.amplitude_destinations:
                if destination.target_sector_id != target.representative_id:
                    continue
                physics_helicity = destination_helicities[
                    destination.destination_id
                ]
                helicity_position = helicity_positions.get(physics_helicity)
                if helicity_position is None:
                    continue
                helicity = helicity_records[physics_helicity]
                if (
                    helicity.get("computed") is not True
                    or helicity.get("structural_zero") is True
                ):
                    continue
                helicity_weight = _decimal(
                    helicity.get("coefficient", 1),
                    "helicity coefficient",
                )
                amplitude = amplitudes[destination.destination_id]
                point_values[helicity_position][color_position] += (
                    normalization
                    * color_weight
                    * helicity_weight
                    * (
                        amplitude[0] * amplitude[0]
                        + amplitude[1] * amplitude[1]
                    )
                )

    def _evaluate_union_resolved_point(
        self,
        point: object,
        selected_helicities: Sequence[int],
        selected_colors: Sequence[int],
        helicity_records: Sequence[dict[str, object]],
        color_records: Sequence[dict[str, object]],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        working_precision: int,
        normalization: Decimal,
        point_values: list[list[Decimal]],
    ) -> None:
        for helicity_position, physics_helicity in enumerate(selected_helicities):
            helicity_record = helicity_records[physics_helicity]
            if (
                helicity_record.get("computed") is not True
                or helicity_record.get("structural_zero") is True
            ):
                continue
            direct_helicity = self._union_helicity_by_physics[physics_helicity]
            if direct_helicity is None:
                raise ArtifactError(
                    "all-flow-union plan is missing a retained physical helicity"
                )
            amplitudes = _evaluate_union_point(
                self._plan,
                cast(Any, point),
                direct_helicity,
                prepared_parameters,
                working_precision,
            )
            helicity_weight = _decimal(
                helicity_record.get("coefficient", 1),
                "helicity coefficient",
            )
            for color_position, color_index in enumerate(selected_colors):
                destination = self._union_destination_by_color[color_index]
                color_weight = _decimal(
                    color_records[color_index].get("coefficient", 1),
                    "color coefficient",
                )
                amplitude = amplitudes[destination.destination_id]
                point_values[helicity_position][color_position] += (
                    normalization
                    * color_weight
                    * helicity_weight
                    * (
                        amplitude[0] * amplitude[0]
                        + amplitude[1] * amplitude[1]
                    )
                )

    def _replay_targets_by_color(self) -> tuple[_ReplayTarget, ...]:
        by_public_id = {
            target.public_flow_id: target
            for target in self._plan.sections.replay_targets
        }
        if len(by_public_id) != len(self._plan.sections.replay_targets):
            raise ArtifactError("recurrence replay targets repeat a public flow")
        try:
            return tuple(
                by_public_id[public_id]
                for public_id in self._plan.sections.public_flow_ids
            )
        except KeyError as exc:
            raise ArtifactError(
                "recurrence public color axis references an absent replay target"
            ) from exc

    def _destination_helicity_maps(self) -> tuple[tuple[int, ...], ...]:
        sections = self._plan.sections
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        physics_by_values = {
            tuple(
                _signed_integer(component, "physics helicity component")
                for component in _sequence(
                    record.get("values"), "physics helicity values"
                )
            ): index
            for index, record in enumerate(helicity_records)
        }
        direct_vectors = {}
        for descriptor in sections.resolved_helicities:
            start = descriptor.public_helicity_start
            stop = start + descriptor.public_helicity_count
            vector = sections.public_helicities[start:stop]
            if len(vector) != sections.external_source_count:
                raise ArtifactError(
                    "recurrence direct helicity has invalid source coverage"
                )
            direct_vectors[descriptor.helicity_id] = vector

        result = []
        for target in self._replay_by_color:
            start = target.source_permutation_start
            stop = start + target.source_permutation_count
            permutation = sections.source_permutations[start:stop]
            if sorted(permutation) != list(range(sections.external_source_count)):
                raise ArtifactError("recurrence replay source mapping is not bijective")
            destinations = [DIRECT_NONE_U32] * sections.amplitude_destination_count
            for destination in sections.amplitude_destinations:
                direct_id = destination.target_helicity_id
                if direct_id == DIRECT_NONE_U32:
                    raise ArtifactError(
                        "topology-replay destination has no resolved helicity"
                    )
                representative = direct_vectors.get(direct_id)
                if representative is None:
                    raise ArtifactError(
                        "recurrence destination references an absent direct helicity"
                    )
                public = [0] * sections.external_source_count
                for representative_slot, helicity in enumerate(representative):
                    public[permutation[representative_slot]] = helicity
                try:
                    destinations[destination.destination_id] = physics_by_values[
                        tuple(public)
                    ]
                except KeyError as exc:
                    raise ArtifactError(
                        "recurrence replay maps a helicity outside public coverage"
                    ) from exc
            if any(value == DIRECT_NONE_U32 for value in destinations):
                raise ArtifactError(
                    "recurrence amplitude destination helicity map is incomplete"
                )
            result.append(tuple(destinations))
        return tuple(result)

    def _union_destinations_by_color(
        self,
    ) -> tuple[_AmplitudeDestination, ...]:
        sections = self._plan.sections
        by_sector = {
            destination.target_sector_id: destination
            for destination in sections.amplitude_destinations
        }
        if len(by_sector) != len(sections.amplitude_destinations):
            raise ArtifactError(
                "all-flow-union repeats an amplitude destination sector"
            )
        try:
            return tuple(
                by_sector[sector_id] for sector_id in sections.public_flow_ids
            )
        except KeyError as exc:
            raise ArtifactError(
                "all-flow-union public color axis references an absent destination"
            ) from exc

    def _union_helicities_by_physics(
        self,
    ) -> tuple[_ResolvedHelicity | None, ...]:
        sections = self._plan.sections
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        physics_by_values = {
            tuple(
                _signed_integer(component, "physics helicity component")
                for component in _sequence(
                    record.get("values"), "physics helicity values"
                )
            ): index
            for index, record in enumerate(helicity_records)
        }
        result: list[_ResolvedHelicity | None] = [None] * len(helicity_records)
        for descriptor in sections.resolved_helicities:
            start = descriptor.public_helicity_start
            stop = start + descriptor.public_helicity_count
            vector = tuple(sections.public_helicities[start:stop])
            if len(vector) != sections.external_source_count:
                raise ArtifactError(
                    "all-flow-union helicity has incomplete source coverage"
                )
            try:
                physics_index = physics_by_values[vector]
            except KeyError as exc:
                raise ArtifactError(
                    "all-flow-union helicity is outside public coverage"
                ) from exc
            if result[physics_index] is not None:
                raise ArtifactError(
                    "all-flow-union repeats a public helicity"
                )
            result[physics_index] = descriptor
        for index, (record, descriptor) in enumerate(
            zip(helicity_records, result, strict=True)
        ):
            if (
                record.get("computed") is True
                and record.get("structural_zero") is not True
                and descriptor is None
            ):
                raise ArtifactError(
                    f"all-flow-union omits computed helicity {index}"
                )
        return tuple(result)


def _signed_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{context} must be an integer")
    return value


__all__ = ["RecurrenceExactExecutor"]
