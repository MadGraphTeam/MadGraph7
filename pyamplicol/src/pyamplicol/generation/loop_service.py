# SPDX-License-Identifier: 0BSD
"""Public source-model loop generation using the shared exact execution plan.

The service owns request/configuration transport. Existing source, graph,
counterterm and reduction boundaries retain responsibility for physics support.
"""

from __future__ import annotations

import cmath
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
from pyamplicol.loops.source_color_declaration import (
    prepare_declared_source_color,
    source_color_fixed_parameters,
)
from pyamplicol.models.loop_color_declaration import resolve_loop_color_declaration
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
    # These are tree evaluator settings. The loop compiler currently selects
    # its existing native kernels/schedules and cannot apply these overrides.
    # Resource cores remain shared and are resolved by the resource policy.
    defaults = type(run.evaluator)()
    ignored = []
    for group, names in (
        ("jit", ("optimization_level", "compress")),
        (
            "optimization",
            (
                "horner_iterations",
                "cpe_iterations",
                "max_horner_variables",
                "max_common_pair_cache_entries",
                "max_common_pair_distance",
                "collect_factors",
            ),
        ),
        ("recurrence", ("point_tile_size", "workspace_mib")),
    ):
        for name in names:
            actual = getattr(getattr(run.evaluator, group), name)
            expected = getattr(getattr(defaults, group), name)
            if (group, name) == ("jit", "compress"):
                actual = run.evaluator.resolved_jit_compress
                expected = defaults.resolved_jit_compress
            if actual != expected:
                ignored.append(f"evaluator.{group}.{name}")
    if run.evaluator.output_chunk_size != defaults.output_chunk_size:
        ignored.append("evaluator.output_chunk_size")
    if ignored:
        reasons.append(
            "public loop generation does not apply nondefault tree evaluator "
            "tuning: " + ", ".join(ignored) + "; use the existing loop kernel defaults"
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
    source_colors: tuple
    color_declaration: Any = None


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
    color_data = None
    fixed_parameters = None
    if run.color.accuracy in ("lc", "nlc"):
        color_data = resolve_loop_color_declaration(compiled.raw_loop_model)
        if color_data is None:
            raise GenerationError(
                "LC/NLC loop generation requires explicit source colour data in "
                "the prepared model; attach a LoopColorDeclaration with "
                "with_loop_color_declaration. Model names or numerical CA/CF "
                "values do not define a formal-colour continuation"
            )
        if color_data.scheme != declaration.scheme:
            raise GenerationError(
                "source colour continuation and effective loop scheme differ"
            )
        fixed_parameters = source_color_fixed_parameters(
            compiled.raw_loop_model, color_data
        )
    from pyamplicol.models.external import CompiledUFOModel

    resolved = replace(resolved, compiled=compiled, model=CompiledUFOModel(compiled))
    expanded = backend._expand_process_set(processes, resolved, phase)
    backend._apply_symbolica_resource_policy(license_state, process_count=len(expanded))
    cache, prepared, source_colors = {}, [], []
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
            current = prepare_loop_request_model(
                compiled,
                born_order_bounds=born,
                loop_order_bounds=loop,
                fixed_parameters=fixed_parameters,
            )
            color = None
            if color_data is not None:
                current, color = prepare_declared_source_color(
                    current, color_data, scheme=declaration.scheme
                )
            cache[key] = current, color
        current, color = cache[key]
        prepared.append(current)
        source_colors.append(color)
    return _LoopContext(
        resolved,
        declaration,
        expanded,
        tuple(prepared),
        corrections,
        tuple(source_colors),
        color_data,
    )


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
            "color_accuracy": backend._run_config.color.accuracy,
            "source_colour_declared": bool(context and context.color_declaration),
            "formal_colour_process_coverage_checked": False,
            "canonical_born_components_checked": False
            if backend._run_config.color.accuracy == "lc"
            else None,
            "correlations": False,
            "resolved": True,
            "api_bundle": True,
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


def _check_correlators(backend, correlators):
    if correlators is None:
        return
    from pyamplicol.correlators import CorrelatorConfig

    if not isinstance(correlators, CorrelatorConfig):
        raise TypeError("correlators must be a CorrelatorConfig")
    if correlators.spin_correlations:
        raise GenerationError(
            "public virtual spin-source generation is not yet supported"
        )
    if backend._run_config.color.accuracy != "full":
        raise GenerationError("virtual colour histories currently require full colour")


def _prepare_directed(program, declarations):
    """Compile existing typed histories over the same retained source currents."""
    from pyamplicol.color.connections import ColorConnection, ColorLeg
    from pyamplicol.loops.correlation_labels import native_color_history
    from pyamplicol.loops.physical_correlations import directed_color_legs
    from pyamplicol.loops.prepared_correlations import (
        prepare_directed_bare,
        prepare_directed_model,
    )

    legs = directed_color_legs(program.bare)
    public_legs = tuple(ColorLeg(leg.label + 1, leg.representation) for leg in legs)
    requests = declarations._resolve_color(public_legs).color_requests
    native_requests = tuple(native_color_history(request) for request in requests)
    histories = tuple(
        (ColorConnection(legs, request.bra), ColorConnection(legs, request.ket))
        for request in native_requests
    )
    prepared = prepare_directed_model(
        program, prepare_directed_bare(program.bare, histories)
    )
    return prepared, native_requests


def generate_loop_artifact(
    backend,
    processes,
    output: os.PathLike[str] | str,
    *,
    model=None,
    mode="error",
    correlators=None,
):
    """Compile source-owned complete model-R2 instructions and publish atomically."""
    _check_correlators(backend, correlators)
    blockers = _feature_blockers(backend, processes)
    if blockers:
        raise GenerationError("; ".join(blockers))
    reporter = GenerationPhaseReporter(backend._progress)
    reporter.start_generation(
        total=3, details={"execution_mode": "loop_recurrence", "backend": "JIT"}
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
            validation_values = {}
            with reporter.phase(
                "loop-compile",
                "Compiling complete loop instructions",
                total=len(context.expanded),
            ) as phase:
                for index, (prepared, entry, source_color) in enumerate(
                    zip(
                        context.prepared,
                        context.expanded,
                        context.source_colors,
                        strict=True,
                    )
                ):
                    program = prepared.compile(
                        entry.process_ir.initial_pdgs,
                        entry.process_ir.final_pdgs,
                        selected_corrections=context.corrections,
                        reduction_engine="fast",
                        coefficient_engine="jit",
                        source_color=source_color,
                        color_accuracy=backend._run_config.color.accuracy,
                    )
                    path = Path(temporary) / f"{entry.request.name}.pacbin"
                    directed, histories = (
                        (None, ())
                        if correlators is None
                        else _prepare_directed(program.virtual, correlators)
                    )
                    write_loop_pacbin(
                        program.virtual,
                        path,
                        resolved=directed is None,
                        directed=directed,
                        directed_histories=histories or None,
                    )
                    validation = backend._generation_config.validation
                    point = build_process_validation_point(
                        entry.process_ir,
                        context.resolved.model,
                        process_id=entry.request.name,
                        seed=validation.seed + index,
                    )
                    if validation.post_build_validation and validation.enabled:
                        validation_values[entry.request.name] = _validation_values(
                            backend,
                            context,
                            entry,
                            program.virtual,
                            directed=directed,
                            histories=histories,
                        )
                    artifacts.append(
                        LoopProcessArtifact(
                            process_id=entry.request.name,
                            expression=entry.request.expression,
                            external_pdgs=program.source_bare.external_request.external_pdgs,
                            loop_runtime_path=path,
                            color_accuracy=backend._run_config.color.accuracy,
                            metadata=LoopExecutionMetadata(
                                scheme=context.declaration.scheme,
                                orders=program.virtual.bare.order_layout,
                                r2_strategy="model",
                                available_r2_strategies=("model",),
                                arithmetic=("double", "double-double", "arbitrary"),
                                source_owned=True,
                                resolved=directed is None,
                                correlations=directed is not None,
                                color_accuracy=backend._run_config.color.accuracy,
                            ),
                            validation_point=point,
                            generation_filters={
                                "loop_coupling_orders": context.corrections,
                                "born_order_bounds": prepared.scope.born_order_bounds,
                                "loop_order_bounds": prepared.scope.loop_order_bounds,
                                "model_declaration": context.declaration.to_dict(),
                                **(
                                    {"correlators": correlators.to_json_dict()}
                                    if correlators is not None
                                    else {}
                                ),
                            },
                        )
                    )
                    phase.update(index + 1, message=entry.request.name)
            with reporter.phase(
                "artifact-writing",
                "Writing and validating virtual artifact"
                if backend._generation_config.validation.post_build_validation
                else "Writing virtual artifact",
                total=1,
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
                    staged_validation=(
                        lambda staged: _validate_artifact(
                            backend, staged, context, artifacts, validation_values
                        )
                    )
                    if backend._generation_config.validation.post_build_validation
                    else None,
                )
                phase.update(1, message=str(result.output))
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


@dataclass(frozen=True)
class _ValidationValues:
    """Small binary64 execution checks; never an independent physics oracle."""

    momenta: tuple
    scales: tuple[str, ...]
    born: tuple[complex, ...]
    virtual: tuple[complex, ...]
    directed: tuple = ()


def _validation_values(
    backend, context, entry, program, *, directed=None, histories=()
):
    validation = backend._generation_config.validation
    points = tuple(
        build_process_validation_point(
            entry.process_ir,
            context.resolved.model,
            process_id=entry.request.name,
            seed=validation.seed + i,
        )
        for i in range(validation.samples)
    )
    if any(not point.available for point in points):
        raise GenerationError(
            "virtual validation momenta unavailable: "
            + str(next(point.error for point in points if not point.available))
        )
    # Positive incoming invariant: an explicit probe scale, never a runtime
    # default or an inferred renormalization convention.
    scales = []
    for point in points:
        total = [
            sum(p[mu] for p in point.four_vectors[: len(entry.process_ir.initial_pdgs)])
            for mu in range(4)
        ]
        scale = total[0] ** 2 - sum(component**2 for component in total[1:])
        if scale <= 0:
            raise GenerationError(
                "virtual validation requires a positive incoming invariant"
            )
        scales.append(str(scale))
    # The native source path consumes decimal strings in every carrier. Preserve
    # the generated binary64 probe values by their round-trip decimal form and
    # send those same inputs to the staged public runtime.
    momenta = tuple(
        tuple(tuple(str(value) for value in leg) for leg in point.four_vectors)
        for point in points
    )
    expected = program.evaluate_ordered_native(momenta, scales)
    directed_values = ()
    if directed is not None:
        projected = directed.bind(program).evaluate_native(momenta, scales)
        # Slicing validation buffers only: each producer evaluation above is a
        # native batch, and the public consumer below evaluates all histories once.
        widths = (
            len(program.bare.order_layout.born),
            len(program.bare.order_layout.virtual) * 3,
        )
        columns = (projected.born, projected.virtual)
        directed_values = tuple(
            (
                history.id,
                *(
                    tuple(
                        complex(float(re), float(im))
                        for re, im in column.take_columns(
                            range(index * width, (index + 1) * width)
                        ).to_decimal()
                    )
                    for column, width in zip(columns, widths, strict=True)
                ),
            )
            for index, history in enumerate(histories)
        )
    return _ValidationValues(
        momenta,
        tuple(scales),
        tuple(complex(float(re), float(im)) for re, im in expected.born.to_decimal()),
        tuple(
            complex(float(re), float(im)) for re, im in expected.virtual.to_decimal()
        ),
        directed_values,
    )


def _validate_artifact(backend, output, context, artifacts, validation_values):
    from pyamplicol.api import Runtime
    from pyamplicol.api.virtual_results import VirtualEvaluation
    from pyamplicol.artifacts import load_manifest

    manifest = load_manifest(output)
    validation = backend._generation_config.validation
    for entry, artifact in zip(context.expanded, artifacts, strict=True):
        runtime = Runtime.load(output, process=entry.request.name)
        if (
            runtime.execution_mode != "loop_recurrence"
            or runtime.artifact_id != manifest.artifact_id
            or runtime.representative_process_key != entry.request.name
        ):
            raise GenerationError(
                "virtual validation loaded incompatible artifact metadata"
            )
        if not validation.enabled:
            continue
        expected = validation_values[entry.request.name]
        result = runtime.evaluate(expected.momenta, mu_r2=expected.scales)
        if not isinstance(result, VirtualEvaluation) or (
            result.shape
            != (len(expected.momenta), len(artifact.metadata.orders.virtual), 3)
            or result.born_shape
            != (len(expected.momenta), len(artifact.metadata.orders.born))
            or result.sectors != artifact.metadata.orders.virtual
            or result.born_orders
            != tuple(sector.order for sector in artifact.metadata.orders.born)
            or result.color_accuracy != artifact.metadata.color_accuracy
        ):
            raise GenerationError(
                "virtual validation returned incompatible Laurent/order axes"
            )
        _compare_validation_result(
            entry.request.name, result, expected.born, expected.virtual, validation
        )
        if expected.directed:
            from pyamplicol.correlators import CorrelatedRequest

            grouped = runtime.evaluate_correlated_many(
                expected.momenta,
                {label: CorrelatedRequest(label) for label, _, _ in expected.directed},
                mu_r2=expected.scales,
            )
            for label, born, virtual in expected.directed:
                current = grouped[label]
                if (
                    not current.correlated
                    or current.shape != result.shape
                    or current.born_shape != result.born_shape
                    or current.sectors != result.sectors
                    or current.born_orders != result.born_orders
                ):
                    raise GenerationError(
                        "directed validation returned incompatible axes"
                    )
                _compare_validation_result(
                    f"{entry.request.name}/{label}", current, born, virtual, validation
                )


def _compare_validation_result(
    name, result, expected_born, expected_virtual, validation
):
    actual_born = tuple(value for row in result.born_values for value in row)
    actual_virtual = tuple(
        value for row in result.values for sector in row for value in sector
    )
    for label, actual, reference in (
        ("Born", actual_born, expected_born),
        ("virtual", actual_virtual, expected_virtual),
    ):
        if len(actual) != len(reference):
            raise GenerationError(
                f"virtual validation returned incompatible {label} axes"
            )
        for index, (value, target) in enumerate(zip(actual, reference, strict=True)):
            if not cmath.isclose(
                complex(value),
                target,
                rel_tol=validation.relative_tolerance,
                abs_tol=validation.absolute_tolerance,
            ):
                raise GenerationError(
                    f"saved virtual execution differs from source execution for "
                    f"{name!r}, {label} coefficient {index} "
                    f"(absolute difference {abs(complex(value) - target):.3e})"
                )
