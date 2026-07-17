# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from pyamplicol.api.errors import GenerationError, ModelError, PyAmpliColError
from pyamplicol.api.requests import (
    CompiledModel,
    ModelSource,
    ProcessAlias,
    ProcessRequest,
    ProcessSet,
)
from pyamplicol.api.results import GenerationPlan, GenerationResult
from pyamplicol.config import ConfigResolution, GenerationConfig, RunConfig
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
)

from ..color.plan import build_color_plan
from ..models.base import Model
from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from .artifact_writer import (
    CompiledProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from .contracts import RuntimeExpressionSchema, StageCompilationInput
from .dag_algorithms import (
    infer_minimal_coupling_order_limits,
    prune_global_helicity_flip_equivalent_roots,
)
from .dag_compiler import _restrict_color_plan, compile_generic_dag
from .dag_types import GenericDAG
from .progress import GenerationPhaseReporter, PhaseHandle
from .runtime_schema import build_runtime_expression_schema
from .stage_compiler import (
    build_and_write_generic_stage_evaluator_artifacts,
    write_model_parameter_evaluator_artifact,
)
from .validation import ValidationPointRecord, build_validation_point

if TYPE_CHECKING:
    from ..licensing import SymbolicaLicenseState
    from .artifact_writer import ApiBundleHook


_ProcessInput = TypeVar("_ProcessInput")
_ProcessOutput = TypeVar("_ProcessOutput")
_MISSING_PROCESS_RESULT = object()
# Symbolica releases the GIL while retaining process-wide mutable symbol state.
# Keep each complete lowering/compilation transaction atomic across generators.
_SYMBOLICA_MATERIALIZATION_LOCK = Lock()


def _builtin_sm_model() -> Model:
    from ..models import BuiltinSMModel

    return BuiltinSMModel()


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    source: ModelSource
    model: Model | None
    compiled: CompiledModel | None = None
    use_compiled_process_catalog: bool = True


@dataclass(frozen=True, slots=True)
class _ProcessSelection:
    max_color_sectors: int | None = None
    reference_color_order: tuple[int, ...] | None = None
    selected_color_sector_ids: frozenset[int] | None = None
    selected_source_helicities: Mapping[int, int] | None = None


@dataclass(frozen=True, slots=True)
class _ExpandedProcess:
    request: ProcessRequest
    process_ir: CanonicalProcessIR
    aliases: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _DagProcess:
    expanded: _ExpandedProcess
    dag: GenericDAG
    coverage: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CompiledProcess:
    expanded: _ExpandedProcess
    dag: GenericDAG
    coverage: Mapping[str, object]
    filters: Mapping[str, object]
    validation_points: tuple[ValidationPointRecord, ...]


@dataclass(frozen=True, slots=True)
class _EvaluatorProcess:
    compiled: _CompiledProcess
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput


def _map_process_phase(
    items: Sequence[_ProcessInput],
    operation: Callable[[_ProcessInput], _ProcessOutput],
    *,
    executor: ThreadPoolExecutor | None,
    max_in_flight: int,
    phase_name: str,
    item_name: Callable[[_ProcessInput], str],
) -> tuple[_ProcessOutput, ...]:
    """Apply one process phase with bounded submissions and stable output order."""

    if max_in_flight < 1:
        raise ValueError("process phase max_in_flight must be positive")
    if executor is None:
        results: list[_ProcessOutput] = []
        for item in items:
            try:
                results.append(operation(item))
            except Exception as exc:
                raise GenerationError(
                    f"{phase_name} failed for process {item_name(item)!r}: {exc}"
                ) from exc
        return tuple(results)

    ordered: list[_ProcessOutput | object] = [_MISSING_PROCESS_RESULT for _ in items]
    active: dict[Future[_ProcessOutput], int] = {}
    next_index = 0

    def submit_one(index: int) -> None:
        item = items[index]
        try:
            future = executor.submit(operation, item)
        except Exception as exc:
            for pending in active:
                pending.cancel()
            if active:
                wait(tuple(active))
            raise GenerationError(
                f"{phase_name} failed to schedule process {item_name(item)!r}: {exc}"
            ) from exc
        active[future] = index

    while next_index < len(items) and len(active) < max_in_flight:
        submit_one(next_index)
        next_index += 1

    while active:
        done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
        failures: list[tuple[int, Exception]] = []
        for future in done:
            index = active.pop(future)
            try:
                ordered[index] = future.result()
            except Exception as exc:
                failures.append((index, exc))
        if failures:
            for pending in active:
                pending.cancel()
            if active:
                wait(tuple(active))
            index, exc = min(failures, key=lambda failure: failure[0])
            item = items[index]
            raise GenerationError(
                f"{phase_name} failed for process {item_name(item)!r}: {exc}"
            ) from exc
        while next_index < len(items) and len(active) < max_in_flight:
            submit_one(next_index)
            next_index += 1

    if any(result is _MISSING_PROCESS_RESULT for result in ordered):
        raise GenerationError(f"{phase_name} did not produce every process result")
    return tuple(cast(_ProcessOutput, result) for result in ordered)


class GenerationBackend:
    def __init__(
        self,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
        progress: ProgressSink | None,
        *,
        api_bundle_hook: ApiBundleHook | None = None,
    ) -> None:
        self._resource_config = config
        self._configuration = _GenerationConfigProvenance.from_config(config)
        self._config = self._configuration.effective
        self._progress = progress
        self._api_bundle_hook = api_bundle_hook

    def plan(
        self,
        processes: ProcessSet,
        *,
        model: ModelSource | CompiledModel | None = None,
    ) -> GenerationPlan:
        source = model or self._configured_model_source()
        task_id = "generation-plan"
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    "Planning processes",
                    total=len(processes.requests),
                )
            )
        try:
            license_state = self._detect_symbolica_license()
            self._apply_symbolica_resource_policy(license_state)
            resolved_model = self._resolve_model_for_plan(source)
            expanded = self._expand_process_set(
                processes,
                resolved_model,
                PhaseHandle(task_id, self._progress, len(processes.requests)),
            )
            self._apply_symbolica_resource_policy(
                license_state,
                process_count=len(expanded),
            )
            coverage: list[dict[str, object]] = []
            for entry in expanded:
                coverage.append(
                    self._plan_concrete_process(
                        entry.process_ir,
                        model=resolved_model.model,
                    )
                )
            result = GenerationPlan(
                concrete_processes=tuple(entry.request for entry in expanded),
                estimated_coverage={
                    "model_kind": resolved_model.source.kind,
                    "color_accuracy": self._color_accuracy,
                    "process_count": len(expanded),
                    "alias_count": sum(len(entry.aliases) for entry in expanded),
                    "processes": tuple(coverage),
                },
                requested_settings=self._configuration.requested,
                effective_settings=self._config,
                adjustments=self._configuration.adjustments,
                unsupported_features=(),
            )
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(task_id, success=False, message=str(exc))
                )
            if isinstance(exc, (GenerationError, ModelError)):
                raise
            raise GenerationError(str(exc)) from exc
        if self._progress is not None:
            self._progress.emit(ProgressEnd(task_id))
        return result

    def generate(
        self,
        processes: ProcessSet,
        output: os.PathLike[str] | str,
        *,
        model: ModelSource | CompiledModel | None = None,
        mode: str = "error",
    ) -> GenerationResult:
        if mode not in ("error", "append", "replace"):
            raise ValueError("generation mode must be 'error', 'append', or 'replace'")
        write_mode = cast(Literal["error", "append", "replace"], mode)
        reporter = GenerationPhaseReporter(self._progress)
        generation_started = time.perf_counter()
        try:
            license_state = self._detect_symbolica_license()
            self._apply_symbolica_resource_policy(license_state)
            source = model or self._configured_model_source()
            with reporter.phase(
                "model-loading",
                "Loading and compiling model",
                total=1,
            ) as phase:
                resolved_model = self._resolve_model(source)
                artifact_model = self._artifact_model(resolved_model)
                phase.update(1, message=artifact_model.name)
            generation_model = resolved_model.model
            if generation_model is None:
                raise GenerationError("generation model resolution produced no model")

            with reporter.phase(
                "process-expansion",
                "Expanding concrete processes",
                total=len(processes.requests),
            ) as phase:
                expanded = self._expand_process_set(processes, resolved_model, phase)
            self._apply_symbolica_resource_policy(
                license_state,
                process_count=len(expanded),
            )

            with TemporaryDirectory(prefix="pyamplicol-generation-") as temporary:
                temporary_root = Path(temporary)
                worker_count = self._process_worker_count(len(expanded))
                executor = (
                    ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="pyamplicol-generation",
                    )
                    if worker_count > 1
                    else None
                )
                try:
                    with reporter.phase(
                        "dag",
                        "Compiling process DAGs",
                        total=len(expanded),
                    ) as phase:
                        compiled = _map_process_phase(
                            expanded,
                            lambda entry: self._compile_for_generation(
                                entry,
                                generation_model,
                                phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="DAG compilation",
                            item_name=lambda entry: entry.request.name,
                        )

                    indexed_compiled = tuple(enumerate(compiled))
                    with reporter.phase(
                        "warmup-filter",
                        (
                            "Applying structural reductions and preparing "
                            "validation points"
                        ),
                        total=len(compiled),
                    ) as phase:
                        prepared = _map_process_phase(
                            indexed_compiled,
                            lambda indexed: self._prepare_warmup_process(
                                indexed[1],
                                generation_model,
                                index=indexed[0],
                                phase=phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="warmup and validation-point preparation",
                            item_name=lambda indexed: indexed[1].expanded.request.name,
                        )

                    with reporter.phase(
                        "evaluator-construction",
                        "Constructing runtime evaluator schemas",
                        total=len(prepared),
                    ) as phase:
                        evaluators = _map_process_phase(
                            prepared,
                            lambda entry: self._construct_evaluator(
                                entry,
                                generation_model,
                                phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="runtime evaluator-schema construction",
                            item_name=lambda entry: entry.expanded.request.name,
                        )

                    jit_total = sum(
                        _runtime_stage_count(evaluator.runtime_schema) + 1
                        for evaluator in evaluators
                    )
                    with reporter.phase(
                        "jit",
                        "Compiling and materializing stage evaluators",
                        total=jit_total,
                    ) as phase:
                        # Symbolica expressions are created while resolving the model on
                        # this caller thread. Keep materialization on that same thread.
                        # The process-wide lock already made this phase serial; moving
                        # between worker threads can violate backend thread affinity.
                        artifact_processes = _map_process_phase(
                            evaluators,
                            lambda evaluator: self._materialize_evaluator(
                                evaluator,
                                generation_model,
                                temporary_root,
                                phase,
                            ),
                            executor=None,
                            max_in_flight=1,
                            phase_name="evaluator materialization",
                            item_name=lambda evaluator: (
                                evaluator.compiled.expanded.request.name
                            ),
                        )
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)

                with reporter.phase(
                    "artifact-writing",
                    "Writing schema-v3 artifact",
                    total=1,
                ) as phase:
                    write_result = write_schema_v3_artifact(
                        Path(output),
                        mode=write_mode,
                        source=resolved_model.source,
                        compiled_model=artifact_model,
                        configuration=self._configuration,
                        processes=artifact_processes,
                        timings=reporter.timings,
                        api_bundle_hook=self._api_bundle_hook,
                    )
                    phase.update(1, message=str(write_result.output))

            with reporter.phase(
                "validation",
                "Validating generated artifact",
                total=1,
            ) as phase:
                validation = self._generation_config.validation
                if validation.post_build_validation:
                    self._validate_generated_artifact(
                        write_result.output,
                        artifact_processes,
                        validation_points={
                            process.expanded.request.name: process.validation_points
                            for process in prepared
                        },
                        expected_api_bundle_path=write_result.api_bundle_path,
                    )
                    message = (
                        f"schema and {validation.samples} numerical samples"
                        if validation.enabled
                        else "schema, references, hashes, and target"
                    )
                else:
                    message = "post-build validation disabled"
                phase.update(1, message=message)

            reporter.timings["total"] = time.perf_counter() - generation_started
            with reporter.phase(
                "timing",
                "Finalizing generation timing",
                total=1,
            ) as phase:
                phase.update(
                    1,
                    message=(
                        f"total={reporter.timings['total']:.6f}s; "
                        f"phases={len(reporter.timings) - 1}"
                    ),
                )
        except Exception as exc:
            if isinstance(exc, PyAmpliColError):
                raise
            raise GenerationError(str(exc)) from exc

        concrete_requests = tuple(entry.expanded.request for entry in prepared)
        return GenerationResult(
            output=write_result.output,
            processes=ProcessSet(
                requests=concrete_requests,
                aliases=processes.aliases,
            ),
            mode=write_mode,
            files=write_result.files,
        )

    def _artifact_model(self, resolved: _ResolvedModel) -> CompiledModel:
        if resolved.compiled is not None:
            return resolved.compiled
        from ..models.loading import compile_model_source

        return compile_model_source("BUILTIN_SM", use_cache=False)

    def _expand_process_set(
        self,
        processes: ProcessSet,
        resolved: _ResolvedModel,
        phase: PhaseHandle,
    ) -> tuple[_ExpandedProcess, ...]:
        aliases_by_target: dict[str, list[ProcessAlias]] = {}
        for alias in processes.aliases:
            aliases_by_target.setdefault(alias.process_name, []).append(alias)
        result: list[_ExpandedProcess] = []
        names: set[str] = set()
        for request_index, request in enumerate(processes.requests, start=1):
            expanded = self._expand_request(request, resolved)
            request_aliases = aliases_by_target.get(request.name, [])
            if request_aliases and len(expanded) != 1:
                raise GenerationError(
                    f"process aliases for {request.name!r} are ambiguous after "
                    f"expansion into {len(expanded)} concrete processes"
                )
            for concrete_index, process_ir in enumerate(expanded, start=1):
                name = _expanded_name(
                    request.name,
                    concrete_index,
                    len(expanded),
                )
                if name in names:
                    raise GenerationError(f"expanded process ID is not unique: {name}")
                names.add(name)
                concrete = ProcessRequest.parse(process_ir.process, name=name)
                alias_records: list[Mapping[str, object]] = []
                for alias in request_aliases:
                    permutation = tuple(alias.particle_permutation) or tuple(
                        range(len(process_ir.legs))
                    )
                    if len(permutation) != len(process_ir.legs):
                        raise GenerationError(
                            f"alias {alias.name!r} permutation has length "
                            f"{len(permutation)}, expected {len(process_ir.legs)}"
                        )
                    if permutation[:2] != (0, 1):
                        raise GenerationError(
                            f"alias {alias.name!r} may only permute final-state "
                            "particles; genuine crossing reuse is not enabled"
                        )
                    alias_expression, alias_pdgs = _permuted_process_identity(
                        process_ir,
                        permutation,
                    )
                    alias_records.append(
                        {
                            "id": alias.name,
                            "expression": alias_expression,
                            "external_pdgs": list(alias_pdgs),
                            "external_permutation": list(permutation),
                        }
                    )
                result.append(
                    _ExpandedProcess(
                        request=concrete,
                        process_ir=process_ir,
                        aliases=tuple(alias_records),
                    )
                )
            phase.update(request_index, message=request.name)
        return tuple(result)

    def _compile_for_generation(
        self,
        expanded: _ExpandedProcess,
        model: Model,
        phase: PhaseHandle,
    ) -> _DagProcess:
        dag, coverage = self._compile_concrete_process(expanded.process_ir, model)
        phase.advance(message=expanded.request.name)
        return _DagProcess(expanded=expanded, dag=dag, coverage=coverage)

    def _prepare_warmup_process(
        self,
        process: _DagProcess,
        model: Model,
        *,
        index: int,
        phase: PhaseHandle,
    ) -> _CompiledProcess:
        reduced = prune_global_helicity_flip_equivalent_roots(process.dag, model)
        validation = self._generation_config.validation
        filters: dict[str, object] = {
            "structural_helicity_reduction": {
                "applied": reduced is not process.dag,
                "before_amplitude_roots": len(process.dag.amplitude_roots),
                "after_amplitude_roots": len(reduced.amplitude_roots),
                "mode": "proven global-helicity-flip equivalence",
            },
        }
        sample_count = (
            validation.samples
            if validation.enabled and validation.post_build_validation
            else 1
        )
        points = tuple(
            build_validation_point(
                reduced,
                model,
                process_id=process.expanded.request.name,
                seed=validation.seed + index * sample_count + sample_index,
            )
            for sample_index in range(sample_count)
        )
        point = points[0]
        phase.advance(
            message=(
                process.expanded.request.name
                if point.available
                else f"{process.expanded.request.name}: metadata-only validation"
            )
        )
        coverage = {
            **dict(process.coverage),
            "current_count": len(reduced.currents),
            "interaction_count": len(reduced.interactions),
            "interaction_evaluation_count": reduced.interaction_evaluation_count,
            "amplitude_root_count": len(reduced.amplitude_roots),
        }
        return _CompiledProcess(
            expanded=process.expanded,
            dag=reduced,
            coverage=coverage,
            filters=filters,
            validation_points=points,
        )

    def _construct_evaluator(
        self,
        process: _CompiledProcess,
        model: Model,
        phase: PhaseHandle,
    ) -> _EvaluatorProcess:
        schema = build_runtime_expression_schema(
            process.dag,
            model,
            process_id=process.expanded.request.name,
        )
        phase.advance(message=process.expanded.request.name)
        return _EvaluatorProcess(
            compiled=process,
            runtime_schema=schema,
            stage_input=StageCompilationInput(process.dag, model, schema),
        )

    def _materialize_evaluator(
        self,
        process: _EvaluatorProcess,
        model: Model,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> CompiledProcessArtifact:
        with _SYMBOLICA_MATERIALIZATION_LOCK:
            return self._materialize_evaluator_unlocked(
                process,
                model,
                temporary_root,
                phase,
            )

    def _materialize_evaluator_unlocked(
        self,
        process: _EvaluatorProcess,
        model: Model,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> CompiledProcessArtifact:
        process_id = process.compiled.expanded.request.name
        evaluator_root = temporary_root / process_id

        def evaluator_progress(event: dict[str, object]) -> None:
            if event.get("stage") == "stage complete":
                phase.advance(message=str(event.get("item", process_id)))

        _blueprint, stage_manifest = build_and_write_generic_stage_evaluator_artifacts(
            process.stage_input,
            process.runtime_schema.to_mapping(),
            evaluator_root,
            model=model,
            stage_local_parameter_layout=True,
            symbolica_settings=self._symbolica_settings(),
            jit_compile=True,
            evaluator_progress_callback=evaluator_progress,
        )
        if not bool(stage_manifest.get("runtime_available")):
            raise GenerationError(
                f"stage evaluator for {process_id!r} is not runtime-available"
            )
        model_parameter_evaluator = write_model_parameter_evaluator_artifact(
            model,
            process.runtime_schema.to_mapping(),
            evaluator_root,
            symbolica_settings=self._symbolica_settings(),
            jit_compile=True,
        )
        phase.advance(message=f"{process_id}: model parameters")
        ir = process.compiled.expanded.process_ir
        dag = process.compiled.dag
        return CompiledProcessArtifact(
            process_id=process_id,
            expression=ir.process,
            color_accuracy=ir.color_accuracy,
            external_pdgs=(*ir.initial_pdgs, *ir.final_pdgs),
            aliases=process.compiled.expanded.aliases,
            runtime_schema=process.runtime_schema,
            stage_manifest=stage_manifest,
            model_parameter_evaluator=model_parameter_evaluator,
            dag_summary={
                "current_count": len(dag.currents),
                "source_count": len(dag.sources),
                "interaction_count": len(dag.interactions),
                "amplitude_root_count": len(dag.amplitude_roots),
                "truncated": False,
            },
            evaluator_root=evaluator_root,
            validation_point=process.compiled.validation_points[0],
            generation_filters=process.compiled.filters,
        )

    def _process_worker_count(self, process_count: int) -> int:
        if process_count < 1:
            return 1
        configured = self._generation_config.workers
        requested = (
            max(1, os.cpu_count() or 1)
            if configured == "auto"
            else max(1, int(configured))
        )
        return min(process_count, requested)

    def _detect_symbolica_license(self) -> SymbolicaLicenseState:
        from ..licensing import detect_symbolica_license

        run = self._run_config
        return detect_symbolica_license(
            suggest=True if run is None else run.symbolica.suggest_license,
            json_mode=False if run is None else str(run.output.format) == "json",
        )

    def _apply_symbolica_resource_policy(
        self,
        state: SymbolicaLicenseState,
        *,
        process_count: int | None = None,
    ) -> None:
        from ..licensing import resolve_symbolica_resource_config

        resolution = resolve_symbolica_resource_config(
            self._resource_config,
            state,
            process_count=process_count,
        )
        self._resource_config = resolution
        self._configuration = _GenerationConfigProvenance.from_config(resolution)
        self._config = resolution.effective

    def _symbolica_settings(self) -> object:
        from ..evaluators.symbolica import SymbolicaEvaluatorSettings

        run = self._run_config
        if run is None:
            return SymbolicaEvaluatorSettings()
        optimization = run.evaluator.optimization
        if optimization.horner_iterations < 1:
            raise GenerationError(
                "evaluator.optimization.horner_iterations must be positive "
                "for generation"
            )
        backend = str(run.evaluator.backend)
        evaluator_backend = "jit" if backend == "jit" else "compiled-complex"
        cores = (
            max(1, os.cpu_count() or 1)
            if optimization.cores == "auto"
            else int(optimization.cores)
        )
        collect_factors = (
            False
            if optimization.collect_factors == "auto"
            else bool(optimization.collect_factors)
        )
        cpp_level = _cpp_optimization_level(run.evaluator.cpp.optimization)
        return SymbolicaEvaluatorSettings(
            backend=evaluator_backend,
            iterations=optimization.horner_iterations,
            cpe_iterations=optimization.cpe_iterations,
            n_cores=cores,
            jit_direct_translation=False,
            jit_optimization_level=run.evaluator.jit.optimization_level,
            max_horner_scheme_variables=optimization.max_horner_variables,
            max_common_pair_cache_entries=(optimization.max_common_pair_cache_entries),
            max_common_pair_distance=optimization.max_common_pair_distance,
            collect_factors=collect_factors,
            compiled_inline_asm="default" if backend == "asm" else "none",
            compiled_optimization_level=cpp_level,
            compiled_native=run.evaluator.cpp.native_arch,
            compiler_path=run.evaluator.cpp.compiler,
            compiler_flags=run.evaluator.cpp.extra_flags,
            compiled_output_chunk_size=run.evaluator.output_chunk_size,
            output_chunk_strategy="uniform",
            output_chunk_autotune_batch_size=run.evaluator.batch_size,
            compiled_chunk_compile_workers=1,
        )

    def _validate_generated_artifact(
        self,
        output: Path,
        processes: Sequence[CompiledProcessArtifact],
        *,
        validation_points: Mapping[str, Sequence[ValidationPointRecord]],
        expected_api_bundle_path: str | None,
    ) -> None:
        import cmath

        from pyamplicol.api import Runtime
        from pyamplicol.artifacts import load_manifest

        manifest = load_manifest(output)
        expected_ids = {process.process_id for process in processes}
        actual_ids = {str(process["id"]) for process in manifest.processes}
        if not expected_ids.issubset(actual_ids):
            raise GenerationError("generated artifact omitted concrete processes")
        by_path = {record.path: record for record in manifest.payloads}
        for process in processes:
            prefix = f"processes/{process.process_id}"
            required = {
                f"{prefix}/physics.json",
                f"{prefix}/execution.json",
                f"{prefix}/validation-momenta.json",
            }
            if not required.issubset(by_path):
                raise GenerationError(
                    f"artifact payload set is incomplete for {process.process_id!r}"
                )
            physics = json.loads((output / f"{prefix}/physics.json").read_text())
            if (
                physics.get("kind") != "pyamplicol-resolved-physics"
                or physics.get("process_id") != process.process_id
            ):
                raise GenerationError(
                    f"runtime physics identity is invalid for {process.process_id!r}"
                )
            execution = json.loads((output / f"{prefix}/execution.json").read_text())
            compiled = execution.get("compiled")
            if not isinstance(compiled, Mapping) or not bool(
                compiled.get("runtime_available")
            ):
                raise GenerationError(
                    f"runtime evaluator is unavailable for {process.process_id!r}"
                )
            runtime = Runtime.load(output, process=process.process_id)
            if runtime.physics.process_id != process.process_id:
                raise GenerationError(
                    f"Rusticol selected the wrong process for {process.process_id!r}"
                )
            validation = self._generation_config.validation
            samples = tuple(
                point.four_vectors
                for point in validation_points.get(process.process_id, ())
                if point.available
            )
            if validation.enabled and samples:
                total = runtime.evaluate(samples)
                resolved_total = runtime.evaluate_resolved(samples).total()
                if len(total) != len(samples) or len(resolved_total) != len(samples):
                    raise GenerationError(
                        f"Rusticol returned an invalid validation shape for "
                        f"{process.process_id!r}"
                    )
                for sample_index, (summed, resolved) in enumerate(
                    zip(total, resolved_total, strict=True),
                    start=1,
                ):
                    difference = abs(complex(summed) - complex(resolved))
                    if not cmath.isclose(
                        complex(summed),
                        complex(resolved),
                        rel_tol=validation.relative_tolerance,
                        abs_tol=validation.absolute_tolerance,
                    ):
                        raise GenerationError(
                            "resolved Rusticol validation does not reduce to the "
                            f"total for {process.process_id!r} sample {sample_index} "
                            f"(absolute difference {difference:.3e})"
                        )
        actual_api_bundle_path = manifest.runtime.get("api_bundle_path")
        if actual_api_bundle_path != expected_api_bundle_path:
            raise GenerationError("artifact API bundle outcome is inconsistent")

    @property
    def _run_config(self) -> RunConfig | None:
        return self._config if isinstance(self._config, RunConfig) else None

    @property
    def _generation_config(self) -> GenerationConfig:
        config = self._config
        return config.generation if isinstance(config, RunConfig) else config

    @property
    def _builtin_process_options(self):
        from ..models.builtin.process_types import BuiltinProcessOptions

        run = self._run_config
        if run is None:
            return BuiltinProcessOptions()
        max_lines = run.process.max_quark_lines
        return BuiltinProcessOptions(
            flavour_scheme=run.process.flavor_scheme,
            include_3qqbar=max_lines is None or max_lines >= 3,
        )

    @property
    def _max_quark_pairs(self) -> int | None:
        run = self._run_config
        return None if run is None else run.process.max_quark_lines

    @property
    def _color_accuracy(self) -> str:
        run = self._run_config
        return "lc" if run is None else run.color.accuracy

    @property
    def _coupling_order_limits(self) -> dict[str, int]:
        run = self._run_config
        if run is None:
            return {}
        return {
            str(name).upper(): int(value)
            for name, value in run.process.max_coupling_orders.items()
        }

    @property
    def _process_selection(self) -> _ProcessSelection:
        run = self._run_config
        if run is None:
            return _ProcessSelection()
        process = run.process
        return _ProcessSelection(
            max_color_sectors=process.max_color_sectors,
            reference_color_order=(
                tuple(int(label) for label in process.reference_color_order)
                if process.reference_color_order
                else None
            ),
            selected_color_sector_ids=(
                frozenset(
                    int(sector_id)
                    for sector_id in process.selected_color_sector_ids
                )
                if process.selected_color_sector_ids
                else None
            ),
            selected_source_helicities=(
                {
                    int(label): int(helicity)
                    for label, helicity in process.selected_source_helicities.items()
                }
                if process.selected_source_helicities
                else None
            ),
        )

    def _configured_model_source(self) -> ModelSource:
        run = self._run_config
        if run is None:
            return ModelSource.built_in_sm()
        return ModelSource.from_config(run.model)

    def _resolve_model_for_plan(
        self,
        source: ModelSource | CompiledModel,
    ) -> _ResolvedModel:
        if isinstance(source, CompiledModel):
            is_builtin = self._is_builtin_compiled_model(source)
            return _ResolvedModel(
                self._source_for_compiled_model(source),
                _builtin_sm_model() if is_builtin else None,
                source,
                use_compiled_process_catalog=not is_builtin,
            )
        if source.kind == "built-in-sm":
            return _ResolvedModel(source, _builtin_sm_model())
        if source.path is None:
            raise ModelError("external model source has no path")

        from ..models.loading import load_cached_model_source, load_compiled_model

        try:
            if source.kind == "compiled":
                compiled = load_compiled_model(source.path)
            else:
                run = self._run_config
                use_cache = True if run is None else run.model.cache
                if not use_cache:
                    raise ModelError(
                        "dry-run does not compile external model sources and model "
                        "caching is disabled; compile the ModelSource explicitly and "
                        "pass its CompiledModel"
                    )
                cache_dir = None if run is None else run.model.cache_dir
                compiled = load_cached_model_source(
                    source.path,
                    restriction=(
                        "default"
                        if source.restriction is None
                        else str(source.restriction)
                    ),
                    simplify=source.simplify,
                    cache_dir=cache_dir,
                )
                if compiled is None:
                    raise ModelError(
                        "dry-run does not compile external model sources; call "
                        "ModelSource.compile() first and pass the returned "
                        "CompiledModel, or populate the configured model cache"
                    )
        except ModelError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc
        is_builtin = self._is_builtin_compiled_model(compiled)
        return _ResolvedModel(
            source,
            _builtin_sm_model() if is_builtin else None,
            compiled,
            use_compiled_process_catalog=not is_builtin,
        )

    @staticmethod
    def _source_for_compiled_model(compiled: CompiledModel) -> ModelSource:
        return ModelSource(kind="compiled", path=compiled._serialized_path)

    @staticmethod
    def _is_builtin_compiled_model(compiled: CompiledModel) -> bool:
        return compiled.source.get("kind") == "built-in-sm"

    def _resolve_model(
        self,
        source: ModelSource | CompiledModel,
    ) -> _ResolvedModel:
        if isinstance(source, CompiledModel):
            if self._is_builtin_compiled_model(source):
                return _ResolvedModel(
                    self._source_for_compiled_model(source),
                    _builtin_sm_model(),
                    source,
                    use_compiled_process_catalog=False,
                )

            from ..models.external import CompiledUFOModel

            return _ResolvedModel(
                self._source_for_compiled_model(source),
                CompiledUFOModel(source),
                source,
            )
        if source.kind == "built-in-sm":
            return _ResolvedModel(source, _builtin_sm_model())
        if source.path is None:
            raise ModelError("external model source has no path")
        from ..models.external import CompiledUFOModel
        from ..models.loading import compile_model_source, load_compiled_model

        try:
            if source.kind == "compiled":
                compiled = load_compiled_model(source.path)
            else:
                run = self._run_config
                use_cache = True if run is None else run.model.cache
                cache_dir = None if run is None else run.model.cache_dir
                compiled = compile_model_source(
                    source.path,
                    restriction=(
                        "default"
                        if source.restriction is None
                        else str(source.restriction)
                    ),
                    simplify=source.simplify,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc
        if self._is_builtin_compiled_model(compiled):
            return _ResolvedModel(
                source,
                _builtin_sm_model(),
                compiled,
                use_compiled_process_catalog=False,
            )
        return _ResolvedModel(source, CompiledUFOModel(compiled), compiled)

    def _expand_request(
        self,
        request: ProcessRequest,
        resolved_model: _ResolvedModel,
    ) -> tuple[CanonicalProcessIR, ...]:
        if (
            resolved_model.compiled is None
            or not resolved_model.use_compiled_process_catalog
        ):
            from ..models.builtin.process_ir import build_process_ir
            from ..models.builtin.process_selection import (
                enumerate_generic_process_set,
            )

            enumeration = enumerate_generic_process_set(
                request.expression,
                self._builtin_process_options,
                max_quark_pairs=self._max_quark_pairs,
            )
            return tuple(
                build_process_ir(
                    entry.process,
                    color_accuracy=self._color_accuracy,
                    options=self._builtin_process_options,
                )
                for entry in enumeration.entries
            )

        from ..processes.model import (
            ModelParticleCatalog,
            build_model_process_ir,
            expand_model_processes,
        )

        catalog = ModelParticleCatalog(
            resolved_model.compiled.ir.name,
            resolved_model.compiled.ir.particles,
        )
        run = self._run_config
        multiparticles = None if run is None else run.process.multiparticles
        candidates = tuple(
            build_model_process_ir(
                process,
                resolved_model.compiled.ir,
                color_accuracy=self._color_accuracy,
            )
            for process in expand_model_processes(
                request.expression,
                catalog,
                multiparticles=multiparticles,
            )
        )
        selected, rejected = _select_color_ready_processes(
            candidates,
            color_accuracy=self._color_accuracy,
        )
        if not selected:
            detail = "; ".join(rejected) or "no concrete processes"
            raise GenerationError(
                f"process request {request.expression!r} has no usable color plan: "
                f"{detail}"
            )
        return selected

    def _plan_concrete_process(
        self,
        process: CanonicalProcessIR,
        *,
        model: Model | None,
    ) -> dict[str, object]:
        selection = self._process_selection
        if (
            selection.selected_color_sector_ids is not None
            and self._color_accuracy != "lc"
        ):
            raise GenerationError(
                "process.selected_color_sector_ids is available only for LC generation"
            )
        color_plan = build_color_plan(
            process,
            color_accuracy=self._color_accuracy,
            max_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            fold_trace_reflections=(
                model is not None
                and model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
        color_plan, missing_sector_ids = _restrict_color_plan(
            color_plan,
            selection.selected_color_sector_ids,
        )
        if missing_sector_ids:
            raise GenerationError(
                f"process {process.process!r} did not materialize requested LC "
                "colour sector ids: "
                + ", ".join(str(sector_id) for sector_id in missing_sector_ids)
            )
        if not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise GenerationError(
                f"process {process.process!r} has no usable color plan: {detail}"
            )
        return {
            "key": process.key,
            "process": process.process,
            "external_particle_count": len(process.legs),
            "color_sector_count": color_plan.sector_count,
            "color_coverage": (
                "selected"
                if color_plan.truncated
                or selection.selected_color_sector_ids is not None
                else "complete"
            ),
            "helicity_coverage": (
                "selected"
                if selection.selected_source_helicities is not None
                else "complete"
            ),
            "color_diagnostics": tuple(color_plan.diagnostics),
            "coupling_order_limits": self._coupling_order_limits,
            "dag_compilation_deferred": True,
        }

    def _compile_concrete_process(
        self,
        process: CanonicalProcessIR,
        model: Model,
    ) -> tuple[GenericDAG, dict[str, object]]:
        selection = self._process_selection
        if (
            selection.selected_color_sector_ids is not None
            and self._color_accuracy != "lc"
        ):
            raise GenerationError(
                "process.selected_color_sector_ids is available only for LC generation"
            )
        color_plan = build_color_plan(
            process,
            color_accuracy=self._color_accuracy,
            max_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            fold_trace_reflections=(
                model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
        color_plan, missing_sector_ids = _restrict_color_plan(
            color_plan,
            selection.selected_color_sector_ids,
        )
        if missing_sector_ids:
            raise GenerationError(
                f"process {process.process!r} did not materialize requested LC "
                "colour sector ids: "
                + ", ".join(str(sector_id) for sector_id in missing_sector_ids)
            )
        if not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise GenerationError(
                f"process {process.process!r} has no usable color plan: {detail}"
            )
        run = self._run_config
        limits = self._coupling_order_limits
        if run is not None and run.process.coupling_order_policy == "minimal":
            inferred = infer_minimal_coupling_order_limits(
                process,
                model=model,
                max_coupling_orders=limits or None,
            )
            limits = inferred or limits
        dag = compile_generic_dag(
            process,
            model=model,
            max_color_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            selected_color_sector_ids=selection.selected_color_sector_ids,
            max_coupling_orders=limits or None,
            max_quark_pairs=self._max_quark_pairs,
            selected_source_helicities=selection.selected_source_helicities,
        )
        if dag.truncated:
            raise GenerationError(
                f"process {process.process!r} DAG was unexpectedly truncated"
            )
        if not dag.has_amplitudes:
            raise GenerationError(
                f"process {process.process!r} has no model-supported amplitudes"
            )
        return (
            dag,
            {
                "key": process.key,
                "process": process.process,
                "color_sector_count": dag.color_plan.sector_count,
                "color_coverage": dag.color_coverage,
                "helicity_coverage": dag.helicity_coverage,
                "source_count": len(dag.sources),
                "current_count": len(dag.currents),
                "interaction_count": len(dag.interactions),
                "interaction_evaluation_count": dag.interaction_evaluation_count,
                "amplitude_root_count": len(dag.amplitude_roots),
                "coupling_order_limits": limits,
            },
        )


def _runtime_stage_count(schema: RuntimeExpressionSchema) -> int:
    stages = schema.to_mapping().get("stages")
    if isinstance(stages, str | bytes) or not isinstance(stages, Sequence):
        raise GenerationError("runtime expression schema stages must be a sequence")
    return len(stages)


def _select_color_ready_processes(
    processes: Sequence[CanonicalProcessIR],
    *,
    color_accuracy: str,
) -> tuple[tuple[CanonicalProcessIR, ...], tuple[str, ...]]:
    """Drop structurally impossible children of an external multiparticle request."""

    selected: list[CanonicalProcessIR] = []
    rejected: list[str] = []
    for process in processes:
        color_plan = build_color_plan(
            process,
            color_accuracy=color_accuracy,
        )
        if color_plan.ready_for_requested_colour:
            selected.append(process)
            continue
        detail = "; ".join(color_plan.diagnostics) or "no color sectors"
        rejected.append(f"{process.process}: {detail}")
    return tuple(selected), tuple(rejected)


def _expanded_name(base: str, index: int, count: int) -> str:
    return base if count == 1 else f"{base}_{index}"


def _permuted_process_identity(
    process: CanonicalProcessIR,
    representative_to_alias: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    """Return public process metadata in a final-state alias's external order."""

    legs = process.legs
    if len(representative_to_alias) != len(legs) or sorted(
        representative_to_alias
    ) != list(range(len(legs))):
        raise GenerationError("alias particle permutation is not complete")
    alias_legs: list[ProcessLegIR | None] = [None] * len(legs)
    for representative_index, alias_index in enumerate(representative_to_alias):
        alias_legs[alias_index] = legs[representative_index]
    if any(leg is None for leg in alias_legs):
        raise GenerationError("alias particle permutation is incomplete")
    ordered_legs = tuple(cast(ProcessLegIR, leg) for leg in alias_legs)
    if any(leg.pdg is None for leg in ordered_legs):
        raise GenerationError("process alias has an unresolved external particle")
    initial = " ".join(leg.particle for leg in ordered_legs if leg.is_initial)
    final = " ".join(leg.particle for leg in ordered_legs if leg.is_final)
    if not initial or not final:
        raise GenerationError("process alias must retain initial and final states")
    return (
        f"{initial} > {final}",
        tuple(int(leg.pdg) for leg in ordered_legs if leg.pdg is not None),
    )


def _cpp_optimization_level(value: str) -> int:
    normalized = value.strip().upper()
    if normalized.startswith("-"):
        normalized = normalized[1:]
    if normalized not in {"O0", "O1", "O2", "O3"}:
        raise GenerationError("evaluator.cpp.optimization must be O0, O1, O2, or O3")
    return int(normalized[1])


def create_generator_backend(
    config: GenerationConfig | RunConfig | ConfigResolution | None,
    progress: ProgressSink | None,
    *,
    api_bundle_hook: ApiBundleHook | None = None,
) -> GenerationBackend:
    return GenerationBackend(config, progress, api_bundle_hook=api_bundle_hook)


__all__ = ["GenerationBackend", "create_generator_backend"]
