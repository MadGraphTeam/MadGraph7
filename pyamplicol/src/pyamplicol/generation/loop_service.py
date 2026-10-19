# SPDX-License-Identifier: 0BSD
"""Public source-model loop generation using the shared exact execution plan.

The service owns request/configuration transport. Existing source, graph,
counterterm and reduction boundaries retain responsibility for physics support.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pyamplicol.api.errors import GenerationError, PyAmpliColError
from pyamplicol.api.requests import ProcessSet
from pyamplicol.api.results import GenerationPlan, GenerationResult
from pyamplicol.loops.compiler import prepare_loop_request_model
from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.physical_orders import PhysicalOrderLayout
from pyamplicol.loops.portable import write_loop_pacbin
from pyamplicol.loops.request import prepare_loop_external_request
from pyamplicol.models.loop_declaration import (
    resolve_loop_model_declaration,
    with_loop_model_declaration,
)

from .dag_algorithms import infer_minimal_coupling_order_limits
from .loop_artifact import LoopExecutionMetadata, LoopProcessArtifact
from .progress import GenerationPhaseReporter, PhaseHandle
from .validation import build_process_validation_point


def _feature_blockers(backend, processes):
    run = backend._run_config
    reasons = []
    if run.color.accuracy != "full":
        reasons.append(
            "public loop generation currently requires color.accuracy='full'; "
            "LC/NLC publication is not integrated"
        )
    if run.color.contraction == "symmetric-group-fft":
        reasons.append(
            "public loop generation does not yet publish rectangular FFT "
            "contraction; use color.contraction='direct' or 'auto'"
        )
    selection = backend._process_selection
    if any(
        getattr(selection, name) is not None
        for name in (
            "max_color_sectors",
            "reference_color_order",
            "selected_color_sector_ids",
            "selected_source_helicities",
        )
    ):
        reasons.append(
            "public loop generation currently requires inclusive helicity "
            "and colour selectors"
        )
    if (
        processes.aliases
        or run.process.flavor_blind
        or run.process.analyze_process_groups != "none"
    ):
        reasons.append(
            "public loop artifacts do not yet publish aliases "
            "or flavour/process-group mappings"
        )
    if run.loop.r2_strategy == "direct":
        reasons.append(
            "public source-model generation currently supports model R2 rules; "
            "direct R2 is unavailable"
        )
    if run.generation.emit_api_bundle:
        reasons.append(
            "virtual SDK bundles are not integrated; set "
            "generation.emit_api_bundle=false for the Python/shared native runtime"
        )
    if run.generation.relation_discovery.mode != "off":
        reasons.append(
            "tree numerical current relation discovery "
            "is unavailable for loop generation"
        )
    if run.evaluator.backend != "jit":
        reasons.append(
            "public loop coefficient generation currently supports "
            "evaluator.backend='jit'"
        )
    return tuple(reasons)


def _order_bounds(limits: Mapping[str, int], corrections):
    """Translate public inclusive ceilings into the existing native range syntax."""
    born = {name: (0, value) for name, value in limits.items()}
    loop = {
        name: (0, value + (2 if name in corrections else 0))
        for name, value in limits.items()
    }
    return born, loop


@dataclass
class _LoopContext:
    resolved: Any
    declaration: Any
    expanded: tuple
    prepared: tuple
    corrections: tuple[str, ...]


def _prepare(backend, processes, source, *, planning, phase):
    license_state = backend._detect_symbolica_license()
    backend._apply_symbolica_resource_policy(license_state)
    resolved = (
        backend._resolve_model_for_plan(source)
        if planning
        else backend._resolve_model(source)
    )
    compiled = backend._artifact_model(resolved)
    if compiled.raw_loop_model is None:
        raise GenerationError(
            "loop generation requires a compiled NLO UFO with its exact raw "
            "sidecar; this model contains only tree data"
        )
    run = backend._run_config
    corrections = tuple(run.process.loop_coupling_orders)
    declaration = resolve_loop_model_declaration(
        compiled.raw_loop_model, run.loop.model, selected_corrections=corrections
    )
    declaration.require_declared_completion(strategy="model")
    compiled = with_loop_model_declaration(compiled, declaration)
    from pyamplicol.models.external import CompiledUFOModel

    resolved = replace(resolved, compiled=compiled, model=CompiledUFOModel(compiled))
    expanded = backend._expand_process_set(processes, resolved, phase)
    backend._apply_symbolica_resource_policy(license_state, process_count=len(expanded))
    cache, prepared = {}, []
    for entry in expanded:
        limits = backend._coupling_order_limits
        if run.process.coupling_order_policy == "minimal":
            # The same model hierarchy, reachable Born closures and inclusive
            # envelope as the tree service, before adding virtual corrections.
            inferred = infer_minimal_coupling_order_limits(
                entry.process_ir,
                model=resolved.model,
                max_coupling_orders=limits or None,
            )
            limits = inferred or limits
        key = tuple(sorted(limits.items()))
        if key not in cache:
            born, loop = _order_bounds(limits, corrections)
            cache[key] = prepare_loop_request_model(
                compiled, born_order_bounds=born, loop_order_bounds=loop
            )
        prepared.append(cache[key])
    return _LoopContext(resolved, declaration, expanded, tuple(prepared), corrections)


def _graph_plan(prepared, entry, corrections):
    structure = prepared.structure
    request = prepare_loop_external_request(
        structure.native, entry.process_ir.initial_pdgs, entry.process_ir.final_pdgs
    )
    process = build_loop_process(
        prepared.counterterms.process(request.incoming, request.outgoing),
        ordinary_model=structure.native,
        perturbative_species_orders=structure.perturbative_species_orders,
        selected_corrections=corrections,
        perturbative_orders={
            r["name"]: r["fields"].get("perturbative_expansion", 0)
            for r in structure.raw.collections["coupling_orders"]
        },
        born_order_bounds=prepared.scope.born_order_bounds,
        loop_order_bounds=prepared.scope.loop_order_bounds,
    )
    return {
        "key": entry.request.name,
        "process": entry.request.expression,
        "external_particle_count": len(request.external_pdgs),
        "born_graph_count": len(process.born),
        "loop_graph_count": len(process.loops),
        "denominator_family_count": len(process.families),
        "orders": asdict(PhysicalOrderLayout.from_plan(process.orders)),
        "born_order_bounds": prepared.scope.born_order_bounds,
        "loop_order_bounds": prepared.scope.loop_order_bounds,
        "graph_conventions": asdict(process.convention),
        "numerator_compilation_deferred": True,
        "numerator_and_reducer_capabilities_checked": False,
        "normalization": request.to_dict()["normalization"],
    }


def _plan_result(backend, processes, blockers, *, context=None, coverage=()):
    return GenerationPlan(
        concrete_processes=tuple(e.request for e in context.expanded)
        if context
        else processes.requests,
        estimated_coverage={
            "contribution_kind": "virtual",
            "execution_mode": "loop_recurrence",
            "r2_strategy": "model",
            "available_r2_strategies": ("model",),
            "arithmetic": ("double", "double-double", "arbitrary"),
            "epsilon_powers": (0, -1, -2),
            "color_accuracy": "full",
            "correlations": False,
            "resolved": False,
            "api_bundle": False,
            "process_count": len(coverage),
            "processes": tuple(coverage),
            "model_declaration": context.declaration.to_dict() if context else None,
        },
        requested_settings=backend._configuration.requested,
        effective_settings=backend._configuration.effective,
        adjustments=backend._configuration.adjustments,
        unsupported_features=blockers,
    )


def plan_loop_generation(backend, processes, *, model=None):
    """Inspect source/local capabilities and graph orders without numerator JIT."""
    blockers = _feature_blockers(backend, processes)
    if blockers:
        return _plan_result(backend, processes, blockers)
    context, coverage = None, []
    try:
        context = _prepare(
            backend,
            processes,
            model or backend._configured_model_source(),
            planning=True,
            phase=PhaseHandle(
                "loop-generation-plan", backend._progress, len(processes.requests)
            ),
        )
        for prepared, entry in zip(context.prepared, context.expanded, strict=True):
            coverage.append(_graph_plan(prepared, entry, context.corrections))
    except (ValueError, TypeError, RuntimeError, ImportError, PyAmpliColError) as error:
        blockers = (str(error),)
    return _plan_result(
        backend, processes, blockers, context=context, coverage=coverage
    )


def generate_loop_artifact(
    backend, processes, output: os.PathLike[str] | str, *, model=None, mode="error"
):
    """Compile source-owned complete model-R2 instructions and publish atomically."""
    blockers = _feature_blockers(backend, processes)
    if blockers:
        raise GenerationError("; ".join(blockers))
    reporter = GenerationPhaseReporter(backend._progress)
    reporter.start_generation(
        total=4, details={"execution_mode": "loop_recurrence", "backend": "JIT"}
    )
    started = time.perf_counter()
    try:
        with reporter.phase(
            "loop-model",
            "Preparing source loop model and processes",
            total=len(processes.requests),
        ) as phase:
            context = _prepare(
                backend,
                processes,
                model or backend._configured_model_source(),
                planning=False,
                phase=phase,
            )
        with TemporaryDirectory(prefix="pyamplicol-loop-generation-") as temporary:
            artifacts = []
            with reporter.phase(
                "loop-compile",
                "Compiling complete loop instructions",
                total=len(context.expanded),
            ) as phase:
                for index, (prepared, entry) in enumerate(
                    zip(context.prepared, context.expanded, strict=True)
                ):
                    program = prepared.compile(
                        entry.process_ir.initial_pdgs,
                        entry.process_ir.final_pdgs,
                        selected_corrections=context.corrections,
                        reduction_engine="fast",
                        coefficient_engine="jit",
                    )
                    path = Path(temporary) / f"{entry.request.name}.pacbin"
                    write_loop_pacbin(program.virtual, path)
                    validation = backend._generation_config.validation
                    point = build_process_validation_point(
                        entry.process_ir,
                        context.resolved.model,
                        process_id=entry.request.name,
                        seed=validation.seed + index,
                    )
                    artifacts.append(
                        LoopProcessArtifact(
                            process_id=entry.request.name,
                            expression=entry.request.expression,
                            external_pdgs=program.source_bare.external_request.external_pdgs,
                            loop_runtime_path=path,
                            metadata=LoopExecutionMetadata(
                                scheme=context.declaration.scheme,
                                orders=program.virtual.bare.order_layout,
                                r2_strategy="model",
                                available_r2_strategies=("model",),
                                arithmetic=("double", "double-double", "arbitrary"),
                                source_owned=True,
                            ),
                            validation_point=point,
                            generation_filters={
                                "loop_coupling_orders": context.corrections,
                                "born_order_bounds": prepared.scope.born_order_bounds,
                                "loop_order_bounds": prepared.scope.loop_order_bounds,
                                "model_declaration": context.declaration.to_dict(),
                            },
                        )
                    )
                    phase.update(index + 1, message=entry.request.name)
            with reporter.phase(
                "artifact-writing", "Writing virtual artifact", total=1
            ) as phase:
                result = backend._write_artifact(
                    Path(output),
                    mode=mode,
                    source=context.resolved.source,
                    compiled_model=context.resolved.compiled,
                    configuration=backend._configuration,
                    processes=artifacts,
                    timings=reporter.timings,
                    api_bundle_hook=backend._api_bundle_hook,
                )
                phase.update(1, message=str(result.output))
            with reporter.phase(
                "validation", "Validating virtual artifact", total=1
            ) as phase:
                if backend._generation_config.validation.post_build_validation:
                    _validate_artifact(backend, result.output, context, artifacts)
                phase.update(1)
        reporter.timings["total"] = time.perf_counter() - started
    except BaseException as error:
        reporter.finish_generation(success=False, message=str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit, PyAmpliColError)):
            raise
        raise GenerationError(str(error)) from error
    reporter.finish_generation()
    return GenerationResult(
        output=result.output,
        processes=ProcessSet(tuple(e.request for e in context.expanded)),
        mode=mode,
        files=result.files,
    )


def _validate_artifact(backend, output, context, artifacts):
    from pyamplicol.api import Runtime
    from pyamplicol.api.virtual_results import VirtualEvaluation

    validation = backend._generation_config.validation
    for entry, artifact in zip(context.expanded, artifacts, strict=True):
        runtime = Runtime.load(output, process=entry.request.name)
        if runtime.execution_mode != "loop_recurrence":
            raise GenerationError(
                "generated artifact selected a non-virtual execution mode"
            )
        if not validation.enabled:
            continue
        points = [
            build_process_validation_point(
                entry.process_ir,
                context.resolved.model,
                process_id=entry.request.name,
                seed=validation.seed + i,
            )
            for i in range(validation.samples)
        ]
        if any(not point.available for point in points):
            raise GenerationError(
                "virtual validation momenta unavailable: "
                + str(next(point.error for point in points if not point.available))
            )
        # The probe scale is the positive physical incoming invariant. It is not
        # a hidden runtime default or a renormalization-scheme declaration.
        scales = []
        for point in points:
            total = [
                sum(
                    p[mu]
                    for p in point.four_vectors[: len(entry.process_ir.initial_pdgs)]
                )
                for mu in range(4)
            ]
            scales.append(total[0] ** 2 - sum(component**2 for component in total[1:]))
        result = runtime.evaluate(
            [point.four_vectors for point in points], mu_r2=scales
        )
        if not isinstance(result, VirtualEvaluation) or result.shape != (
            len(points),
            len(artifact.metadata.orders.virtual),
            3,
        ):
            raise GenerationError(
                "virtual validation returned incompatible Laurent/order axes"
            )
