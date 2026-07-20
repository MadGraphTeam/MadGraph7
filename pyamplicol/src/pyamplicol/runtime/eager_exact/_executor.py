# SPDX-License-Identifier: 0BSD
"""Public facade for exact replay of a prepared eager process artifact."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.api.results import ResolvedEvaluation
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import confined_path
from pyamplicol.runtime.eager_exact._contracts import (
    _default_kernel_loader,
    _KernelLoader,
    _PayloadIndex,
    _read_json,
    _selected_process,
)
from pyamplicol.runtime.eager_exact._execution import _evaluate_point
from pyamplicol.runtime.eager_exact._plan import _EagerExactPlan
from pyamplicol.runtime.symbolica_exact import (
    _apply_lc_replay_input_mapping,
    _apply_lc_replay_resolved,
    _decimal,
    _lc_replay_plan,
    _prepare_points,
    _reduce_resolved,
    _runtime_state,
    _working_precision,
)


class EagerExactExecutor:
    """Replay one schema-v3 eager process through retained exact kernel states."""

    def __init__(
        self,
        artifact: Path,
        process_id: str,
        native_runtime: Any,
        *,
        kernel_loader: _KernelLoader | None = None,
    ) -> None:
        self._artifact = Path(artifact).expanduser().resolve(strict=True)
        self._native_runtime = native_runtime
        manifest = load_manifest(self._artifact)
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
                f"process {representative_id!r} must declare one eager "
                "execution manifest"
            )
        physics_path = process.get("physics_path")
        if not isinstance(physics_path, str):
            raise ArtifactError(f"process {representative_id!r} has no physics path")
        payloads = _PayloadIndex.from_manifest(manifest)
        payloads.require(
            physics_path, role="runtime-physics", process_id=representative_id
        )
        execution = _read_json(
            confined_path(self._artifact, execution_records[0].path),
            "eager execution metadata",
        )
        physics = _read_json(
            confined_path(self._artifact, physics_path), "runtime physics metadata"
        )
        process_root = self._artifact / "processes" / representative_id
        self._execution = execution
        self._physics = physics
        self._permutation = permutation
        self._lc_replay = _lc_replay_plan(execution, physics, permutation)
        self._plan = _EagerExactPlan.load(
            artifact_root=self._artifact,
            process_root=process_root,
            process_id=representative_id,
            execution=execution,
            manifest=manifest,
            kernel_loader=kernel_loader or _default_kernel_loader,
        )

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
        parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state["model_parameter_values"]
        )
        normalization = _decimal(state["normalization_factor"], "runtime normalization")
        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            exact_parameters = self._plan.resolve_model_parameters(
                parameters,
                working_precision,
            )
            evaluation_points = (
                points
                if self._lc_replay is None
                else tuple(
                    _apply_lc_replay_input_mapping(point, entry.input_mapping)
                    for entry in self._lc_replay.entries
                    for point in points
                )
            )
            amplitudes = tuple(
                _evaluate_point(
                    self._plan,
                    point,
                    exact_parameters.runtime,
                    exact_parameters.prepared,
                    working_precision,
                )
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
                tuple(tuple(+entry for entry in colors) for colors in helicity_values)
                for helicity_values in values
            )
        return ResolvedEvaluation(
            values=values,
            helicity_ids=helicity_ids,
            color_ids=color_ids,
            color_accuracy=cast(Any, str(self._physics["color_accuracy"])),
        )
