# SPDX-License-Identifier: 0BSD
"""Public source-model loop generation through the shared-recursion engine v2.

The service owns request/configuration transport; the v2 generator
(``loops/v2``) publishes plan v4 processes (``loops/v2/artifact.py``; design
§12). v2 is the only one-loop engine: the per-diagram legacy engine was
removed (user ruling 2026-10-28, M-QCD Q3).
"""

from __future__ import annotations

import cmath
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pyamplicol._internal.versions import LOOP_V2_RUNTIME_CAPABILITY
from pyamplicol.api.errors import GenerationError, PyAmpliColError
from pyamplicol.api.requests import ProcessSet
from pyamplicol.api.results import GenerationPlan, GenerationResult
from pyamplicol.config import ColorContraction
from pyamplicol.loops.covariant_model import with_source_repairs
from pyamplicol.loops.model_cms import apply_mass_scheme, require_real_fixed_parameters
from pyamplicol.loops.order_names import CouplingOrderNameError, CouplingOrderNames
from pyamplicol.loops.orders import canonical_order
from pyamplicol.models.loop_color_declaration import (
    resolve_loop_color_declaration,
    source_loop_color_declaration,
)
from pyamplicol.models.loop_declaration import (
    resolve_loop_model_declaration,
    with_loop_model_declaration,
)
from pyamplicol.models.loop_preparation import (
    LoopPreparation,
    resolve_loop_preparation,
    with_loop_preparation,
)

from .dag_algorithms import infer_minimal_coupling_order_limits
from .loop_artifact import LoopV2ExecutionMetadata, LoopV2ProcessArtifact
from .progress import GenerationPhaseReporter, PhaseHandle
from .validation import build_process_validation_point

_LOGGER = logging.getLogger("pyamplicol.generation")

# The binary64 carrier at a staged-validation miss, under the user's O1 rule
# of 2026-10-23 (HANDOFF §0): its virtual is held to max(tolerance, E), E its
# own stability estimate, which must not exceed 1e-9 (or the tolerance, where
# a looser one is configured); its Born to the tolerance. E comes from the
# recommended checks (``loops.v2.stability.RECOMMENDED_CHECKS``: one generic
# exact rotation and the rescaled reduction check), the O1 rule's estimate
# (the reduction direction alone does not see tree-current rounding: 3.1e-13
# against a 1.0e-12 error, d d~ > g g T12; a rotation about the beam axis and
# the cached reduction check missed binary64 errors by 23-96 x, package
# mq-stab).
_BINARY64_ESTIMATE_CAP = 1e-9

# The tree engine's ``[process]`` helicity and colour selectors, which
# leading-colour v2 requests take (``_v2_selection``).
_SELECTORS = (
    "max_color_sectors",
    "reference_color_order",
    "selected_color_sector_ids",
    "selected_source_helicities",
)


def _feature_blockers(backend, processes):
    """The configuration refusals of a one-loop request
    (``refusals.LOOP_CONFIGURATION_REFUSALS``), each a ``ListedBlocker``: its
    listed text, which the dry run shows, with its key."""

    from pyamplicol.loops.v2.refusals import ListedBlocker

    run = backend._run_config
    reasons = []
    if run.resolved_color.contraction != "direct" and run.color.fft_basis != "adjoint":
        reasons.append(
            ListedBlocker(
                "fft-basis-trace",
                "the one-loop seed FFT (design §7.3) acts on the external adjoint "
                "labels; color.fft_basis = 'trace' is unavailable for one-loop "
                "requests (use 'adjoint')",
            )
        )
    selection = backend._process_selection
    if not _leading_colour(run) and any(
        getattr(selection, name) is not None for name in _SELECTORS
    ):
        # Leading-colour requests take them (``_v2_selection``).
        reasons.append(
            ListedBlocker(
                "colour-selectors",
                "public loop generation requires inclusive helicity and colour "
                "selectors at next-to-leading and full colour; leading colour "
                "(color.accuracy='lc') selects them",
            )
        )
    if processes.aliases:
        reasons.append(
            ListedBlocker(
                "process-aliases",
                "public loop artifacts do not yet publish permutation aliases ("
                + ", ".join(sorted(alias.name for alias in processes.aliases))
                + ")",
            )
        )
    if run.process.analyze_process_groups != "none":
        reasons.append(
            ListedBlocker(
                "process-group-analysis",
                "public loop artifacts do not yet publish numerical "
                "process-group analyses (process.analyze_process_groups = "
                f"{str(run.process.analyze_process_groups)!r})",
            )
        )
    if run.loop.r2_strategy == "direct":
        reasons.append(
            ListedBlocker(
                "direct-r2",
                "public source-model generation currently supports model R2 "
                "rules; direct R2 is unavailable",
            )
        )
    if run.generation.relation_discovery.mode != "off":
        reasons.append(
            ListedBlocker(
                "relation-discovery",
                "tree numerical current relation discovery "
                "is unavailable for loop generation",
            )
        )
    if run.evaluator.backend != "jit":
        reasons.append(
            ListedBlocker(
                "evaluator-backend",
                "public loop coefficient generation currently supports "
                "evaluator.backend='jit'",
            )
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
            ListedBlocker(
                "evaluator-tuning",
                "public loop generation does not apply nondefault tree evaluator "
                "tuning: "
                + ", ".join(ignored)
                + "; use the existing loop kernel defaults",
            )
        )
    return tuple(reasons)


@dataclass
class _LoopContext:
    resolved: Any
    declaration: Any
    expanded: tuple
    corrections: tuple[str, ...]
    preparation: LoopPreparation
    color_declaration: Any = None
    limits: tuple[dict[str, int], ...] = ()


def _prepare(
    backend,
    processes,
    source,
    *,
    planning,
    phase,
):
    license_state = backend._detect_symbolica_license()
    backend._apply_symbolica_resource_policy(license_state)
    resolved = (
        backend._resolve_model_for_plan(source)
        if planning
        else backend._resolve_model(source)
    )
    compiled = backend._artifact_model(resolved)
    if compiled.raw_loop_model is None:
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            "tree-only-model",
            "loop generation requires a compiled NLO UFO with its exact raw "
            "sidecar; this model contains only tree data",
        )
    # Structural source repairs (``covariant_model.repair_source_declarations``)
    # on the compiled model the artifact publishes, which records them.
    compiled = with_source_repairs(compiled)
    run = backend._run_config
    # Coupling-order names resolve once, here, against the model's declared
    # names (any case); the loop side carries the declared spelling and the
    # tree engine's keys translate back (``loops.order_names``).
    names = _order_names(compiled)
    corrections = _resolve_orders(
        names, run.process.loop_coupling_orders, "process.loop_coupling_orders"
    )
    configured = run.loop.model
    mass_scheme = None
    if configured is not None and configured.mass_scheme_only:
        # ``[loop.model.scheme] mass_scheme`` alone (MadLoop's set
        # complex_mass_scheme True) selects the mass scheme of the model's own
        # declaration.
        mass_scheme, configured = configured.scheme.mass_scheme, None
    if configured is not None and configured.loop_coupling_orders is not None:
        configured = replace(
            configured,
            loop_coupling_orders=_resolve_orders(
                names,
                configured.loop_coupling_orders,
                "loop.model.loop_coupling_orders",
            ),
        )
    declaration = resolve_loop_model_declaration(
        compiled.raw_loop_model, configured, selected_corrections=corrections
    )
    if mass_scheme is not None:
        declaration = replace(
            declaration, scheme=replace(declaration.scheme, mass_scheme=mass_scheme)
        )
    declaration.require_declared_completion(strategy="model")
    compiled = with_loop_model_declaration(compiled, declaration)
    preparation = resolve_loop_preparation(
        compiled.raw_loop_model, run.loop.preparation
    )
    compiled = with_loop_preparation(compiled, preparation)
    # The mass scheme of the declaration (M-SM MS-CMS): real on-shell, or
    # MG5_aMC's complex-mass model of the source at the card of the fixed
    # parameters (``loops.model_cms``); its refusals are listed, so the dry
    # run reports them.
    compiled = apply_mass_scheme(
        compiled,
        declaration.scheme.mass_scheme,
        fixed_parameters=require_real_fixed_parameters(preparation.fixed_parameters),
        cache_dir=run.model.cache_dir,
    )
    color_data = None
    projected = run.color.accuracy in ("lc", "nlc")
    if projected or run.resolved_color.contraction != "direct":
        # LC and NLC take the shipped declaration of a model that carries
        # none (MQ-NC, ``models.loop_color_declarations``); a full-colour
        # request's contraction choice reads the model's own only, as before.
        color_data = (
            resolve_loop_color_declaration
            if projected
            else source_loop_color_declaration
        )(compiled.raw_loop_model)
        # The FFT detects and certifies its seed orbits on the exact seed
        # cells (``loops.v2.colour.seed_fourier``), so at full colour it
        # needs no formal-colour continuation; LC and NLC do.
        if color_data is None and projected:
            # No declaration covers this source and scheme (the shipped
            # N_c data are scheme-gated, MQ-NC: the QCD+QED models' scheme
            # resolves none).
            from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

            raise LoopRequestUnsupported(
                "colour-declaration-coverage",
                f"{run.color.accuracy.upper()} loop generation requires "
                "explicit source colour data in the prepared model and no "
                "shipped colour declaration covers this model and scheme; "
                "attach a LoopColorDeclaration with "
                "with_loop_color_declaration. Model names or numerical CA/CF "
                "values do not define a formal-colour continuation",
            )
        if color_data is not None and color_data.scheme != declaration.scheme:
            from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

            raise LoopRequestUnsupported(
                "colour-declaration-coverage",
                "source colour continuation and effective loop scheme differ",
            )
    from pyamplicol.models.external import CompiledUFOModel

    resolved = replace(resolved, compiled=compiled, model=CompiledUFOModel(compiled))
    expanded = backend._expand_process_set(processes, resolved, phase)
    backend._apply_symbolica_resource_policy(license_state, process_count=len(expanded))
    entry_limits = []
    explicit = _resolve_limits(names, run.process.max_coupling_orders)
    for entry in expanded:
        limits = explicit
        if run.process.coupling_order_policy == "minimal":
            # The same model hierarchy, reachable Born closures and inclusive
            # envelope as the tree service, before adding virtual corrections.
            inferred = infer_minimal_coupling_order_limits(
                entry.process_ir,
                model=resolved.model,
                max_coupling_orders=limits or None,
            )
            limits = names.mapping(inferred) if inferred else limits
        entry_limits.append(dict(limits))
    return _LoopContext(
        resolved,
        declaration,
        expanded,
        corrections,
        preparation,
        color_data,
        tuple(entry_limits),
    )


def _order_names(compiled) -> CouplingOrderNames:
    """The declared coupling orders of the loop model: its NLO source's (the
    names the request scope matches), else the compiled IR's."""

    collections = getattr(compiled.raw_loop_model, "collections", None)
    try:
        if collections is None:
            return CouplingOrderNames.of_model(compiled)
        return CouplingOrderNames.of(
            record["name"] for record in collections["coupling_orders"]
        )
    except CouplingOrderNameError as error:
        raise GenerationError(str(error)) from error


def _resolve_orders(names, values, what: str) -> tuple[str, ...]:
    """User-supplied coupling-order names in the model's declared spelling
    (sorted, unique), refused when the model does not declare them."""

    try:
        return tuple(sorted(set(names.resolve_all(values, what=what))))
    except CouplingOrderNameError as error:
        raise GenerationError(str(error)) from error


def _resolve_limits(names, limits: Mapping[str, int]) -> dict[str, int]:
    """``process.max_coupling_orders`` keyed by the declared names."""

    out: dict[str, int] = {}
    for name, value in limits.items():
        (declared,) = _resolve_orders(names, (name,), "process.max_coupling_orders")
        if declared in out:
            raise GenerationError(
                f"process.max_coupling_orders names the coupling order "
                f"{declared!r} twice"
            )
        out[declared] = int(value)
    return out


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
            "requested_color_contraction": backend._run_config.color.contraction,
            "rectangular_color_process_coverage_checked": False,
            "source_colour_declared": bool(context and context.color_declaration),
            "formal_colour_process_coverage_checked": False,
            "canonical_born_components_checked": False
            if backend._run_config.color.accuracy == "lc"
            else None,
            "correlations": False,
            "resolved": True,
            "api_bundle": True,
            "process_count": len(coverage),
            "exact_flavor_grouping_requested": backend._run_config.process.flavor_blind,
            "flavor_transport_checked": False,
            "processes": tuple(coverage),
            "model_declaration": context.declaration.to_dict() if context else None,
            "model_preparation": context.preparation.to_dict() if context else None,
            "source_compatibility_diagnostics": (),
        },
        requested_settings=backend._configuration.requested,
        effective_settings=backend._configuration.effective,
        adjustments=backend._configuration.adjustments,
        unsupported_features=blockers,
    )


def plan_loop_generation(backend, processes, *, model=None):
    """The v2 dry run (``_plan_v2``): request and model-preparation checks
    without loop generation or numerator JIT."""
    return _plan_v2(
        backend, processes, _feature_blockers(backend, processes), model=model
    )


def _check_correlators(backend, correlators):
    if correlators is None:
        return
    from pyamplicol.correlators import CorrelatorConfig

    if not isinstance(correlators, CorrelatorConfig):
        raise TypeError("correlators must be a CorrelatorConfig")
    selection = backend._process_selection
    partial = tuple(name for name in _SELECTORS if getattr(selection, name) is not None)
    declared = bool(
        correlators.color_correlations
        or correlators.spin_correlations
        or correlators.all_color_through_order is not None
    )
    if partial and declared and _leading_colour(backend._run_config):
        # Selectors exist at leading colour only (``_feature_blockers``).
        # As the tree engine's correlated generation (``generation.
        # correlated``): a spin source on a leg of fixed helicity would
        # ignore the selection.
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            "selected-correlators",
            "correlated generation requires complete colour and helicity "
            "coverage; remove " + ", ".join(f"process.{name}" for name in partial),
        )
    if declared and backend._run_config.color.contraction == "symmetric-group-fft":
        # v2's own refusal, raised before the model is prepared: the forced
        # seed FFT serves the inclusive virtual only (design §7.3).
        from pyamplicol.loops.v2.observables import FORCED_FFT_OBSERVABLES
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported("forced-seed-fft-unserved", FORCED_FFT_OBSERVABLES)


def _flavor_grouping_outcome(backend, correlators, *, reused=False):
    """Inspection metadata reports reuse separately from the opt-in request."""
    requested = backend._run_config.process.flavor_blind
    checked = requested
    reason = None
    if requested and not reused:
        reason = "no exact identity-port member was admitted"
    return {
        "requested": requested,
        "transport_checked": checked,
        "reuse_applied": reused,
        "independent_reason": reason,
    }


def generate_loop_artifact(
    backend,
    processes,
    output: os.PathLike[str] | str,
    *,
    model=None,
    mode="error",
    correlators=None,
):
    """Generate every process through v2 and publish plan v4 atomically
    (``_generate_v2``)."""
    _check_correlators(backend, correlators)
    return _generate_v2(
        backend, processes, output, model=model, mode=mode, correlators=correlators
    )


# --------------------------------------------------------------------- engine v2

# The seed contraction policy of a v2 request (design §7.3): "auto"
# contracts the certified symmetric-group orbit pairs with the FFT and the
# rest directly, "direct" every weight directly, and the forced FFT requires
# a certified pair (refused at generation otherwise, ``LoopSlice.
# _plan_contraction``).
_V2_CONTRACTION = {
    ColorContraction.AUTO: "auto",
    ColorContraction.DIRECT: "direct",
    ColorContraction.SYMMETRIC_GROUP_FFT: "fft",
}


def _v2_blockers(backend) -> tuple[str, ...]:
    """Configurations the v2 engine does not cover.

    Requests are refused per process, before any loop generation, by
    ``_v2_request`` (no Born amplitude, undeclared corrections, coloured
    widths in the Born currents, an unidentified scale parameter) and
    ``loops.v2.assemble.prepare_request_model`` (the model preparation);
    the generator fails closed with its own reason on what it does not
    admit (e.g. chiral loop-fermion vertices, WP3.2)."""
    run = backend._run_config
    reasons = []
    preparation = run.loop.preparation
    if preparation is not None and preparation.covariant_multiplets:
        from pyamplicol.loops.v2.refusals import ListedBlocker

        reasons.append(
            ListedBlocker(
                "covariant-multiplets",
                "v2 virtuals take no loop.preparation covariant multiplets",
            )
        )
    return tuple(reasons)


def _leading_colour(run) -> bool:
    return run.color.accuracy == "lc"


def _v2_selection(backend, compiled, process: str, label: str):
    """The leading-colour selection of a v2 request (``loops.v2.selection.
    Selection``) from the tree engine's generation selectors, or None when
    they keep every helicity and colour flow.

    ``selected_source_helicities`` fixes leg helicities (labels as the tree
    engine's; an unknown leg or a helicity the leg does not have is refused
    here, so in the dry run too). The colour selectors pick sectors of the
    tree engine's leading-colour plan (``reference_color_order`` first, at most
    ``max_color_sectors``, then ``selected_color_sector_ids`` by id), each a
    flow and, for a folded trace, its reflection (the tree engine's physical
    flows); a missing sector id is refused as the tree engine refuses it.
    The plan options are kept with the selection (``color_order``,
    ``max_sectors``): the artifact numbers its flows in that plan's order,
    as the tree runtime does (``reference_color_order`` alone selects
    nothing and only orders)."""

    from pyamplicol.loops.v2 import selection as lc

    chosen = backend._process_selection
    helicities = tuple(sorted((chosen.selected_source_helicities or {}).items()))
    try:
        lc.require_leg_helicities(compiled, process, helicities)
    except ValueError as error:
        raise GenerationError(f"process {label}: {error}") from error
    order = (
        tuple(int(x) for x in chosen.reference_color_order)
        if chosen.reference_color_order
        else None
    )
    flows = None
    if (
        chosen.max_color_sectors is not None
        or chosen.selected_color_sector_ids is not None
    ):
        plan = lc.leading_colour_plan(
            compiled,
            process,
            max_sectors=chosen.max_color_sectors,
            reference_color_order=order,
        )
        wanted = chosen.selected_color_sector_ids
        known = {int(sector.id) for sector in plan.sectors}
        if wanted is not None and (missing := sorted(set(wanted) - known)):
            raise GenerationError(
                f"process {label} did not materialize requested LC colour sector "
                "ids: " + ", ".join(str(sector_id) for sector_id in missing)
            )
        complete = {
            flow.id
            for flow in lc.plan_flows(lc.leading_colour_plan(compiled, process), ())
        }
        kept = tuple(
            flow
            for flow in lc.plan_flows(plan, ())
            if wanted is None or flow.sector in wanted
        )
        if not kept:
            raise GenerationError(f"process {label} keeps no leading-colour flow")
        if {flow.id for flow in kept} != complete:
            flows = tuple(flow.word for flow in kept)
    if not helicities and flows is None and order is None:
        return None
    # A sector cap orders only the flows of a truncated plan: kept with a
    # colour-flow subset (a cap that keeps every flow keeps the plan).
    return lc.Selection(
        helicities,
        flows,
        order,
        chosen.max_color_sectors if flows is not None else None,
    )


def _blocked(backend, processes) -> tuple[str, ...]:
    return _feature_blockers(backend, processes) + _v2_blockers(backend)


def v2_loop_orders(
    born_orders: Sequence[Sequence[tuple[str, int]]],
    corrections: Sequence[str],
    names: Iterable[str] = (),
) -> dict[str, int] | tuple[dict[str, int], ...]:
    """The exact loop orders of a v2 request: one per Born amplitude order
    and correction, the Born order raised by two powers of that correction
    (a mapping for one, else a sequence in Born-order, then correction
    order: e.g. QCD corrections to QCD and photon Born currents, or QCD and
    QED corrections to one Born order). Every mapping names ``names``, the
    corrections and the Born orders' couplings (zero powers kept)."""

    keys = sorted(
        {*names, *corrections, *(name for order in born_orders for name, _ in order)}
    )
    orders: list[dict[str, int]] = []
    for born in born_orders:
        for correction in corrections:
            loop = dict.fromkeys(keys, 0)
            loop.update(dict(born))
            loop[correction] += 2
            if loop not in orders:
                orders.append(loop)
    if not orders:
        raise GenerationError(
            "a virtual needs at least one Born amplitude order and one correction"
        )
    return orders[0] if len(orders) == 1 else tuple(orders)


def _require_declared_corrections(compiled, corrections) -> None:
    """Every correction must be a perturbative coupling order of the source
    model (``perturbative_expansion`` >= 1). Without it the v2 generator
    would admit no loop species of that order and silently return a
    vanishing correction."""

    orders = {
        record["name"]: record["fields"].get("perturbative_expansion", 0)
        for record in compiled.raw_loop_model.collections["coupling_orders"]
    }
    undeclared = [name for name in corrections if orders.get(name, 0) < 1]
    if undeclared:
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            "undeclared-corrections",
            "requested corrections require declared NLO model support: the "
            f"source model does not expand {undeclared} perturbatively",
        )


def _v2_scale_parameter(compiled, fixed: Mapping[str, str]) -> str:
    """The model's renormalisation-scale parameter: the adjustable external
    parameter at UFO/SLHA code (LOOP, 1), as in NLOCT models and MG5_aMC's
    loop_sm, else the only adjustable external parameter of block LOOP
    (MG5_aMC's loop_qcd_qed_sm keeps MU_R at LOOP 666)."""

    externals = [
        record
        for record in compiled.ir.parameters
        if record.nature == "external"
        and (record.lhablock or "").upper() == "LOOP"
        and record.name not in fixed
    ]
    found = [record.name for record in externals if tuple(record.lhacode) == (1,)]
    if len(found) != 1:
        found = [record.name for record in externals]
    if len(found) != 1:
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            "scale-parameter",
            "the model's renormalisation-scale parameter is not identified: v2 "
            "takes the adjustable external parameter at UFO/SLHA code (LOOP, 1) "
            "or the only one of block LOOP, and this model has "
            f"{sorted(found) or 'none'} there",
        )
    return str(found[0])


def particle_veto_codes(compiled, names: Iterable[str]) -> tuple[int, ...]:
    """The |PDG| codes of ``generation.particle_veto`` (model particle names;
    a name vetoes its antiparticle too)."""

    codes = {}
    for record in compiled.model["particles"]:
        code = abs(int(record["pdg_code"]))
        codes[str(record["name"])] = codes[str(record["antiname"])] = code
    unknown = sorted(set(names) - set(codes))
    if unknown:
        raise GenerationError(
            f"generation.particle_veto names particles the model does not have: "
            f"{unknown}"
        )
    return tuple(sorted({codes[name] for name in names}))


def _loop_orders_record(request) -> object:
    orders = [dict(orders) for orders in request.exact_loop_orders]
    return orders[0] if len(orders) == 1 else orders


def _v2_request(backend, context, index, born_dags=None):
    """The slice request of one expanded process (``loops.v2.assemble``),
    checked before any loop generation; its Born DAG is appended to
    ``born_dags`` when a list is given (the dry run's policy checks read it).

    The Born bounds are the process's coupling-order limits. The loop orders
    follow its Born amplitude orders (``v2_loop_orders``), read from the Born
    DAG (``artifact.born_amplitude_orders``), which must exist and propagate
    no coloured particle with a width (``artifact.require_zero_coloured_widths``;
    the slice re-checks its loop-side tree currents after generation). The
    renormalisation-scale parameter is resolved here
    (``_v2_scale_parameter``)."""
    from pyamplicol.loops.model_cms import (
        require_native_pole_masses,
        require_stable_external_states,
        scheme_metadata,
    )
    from pyamplicol.loops.v2.artifact import (
        born_amplitude_orders,
        require_zero_coloured_widths,
    )
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.ew_policy import correction_orders, require_zero_widths

    run = backend._run_config
    compiled = context.resolved.compiled
    limits = context.limits[index]
    entry = context.expanded[index]
    process = entry.process_ir.process
    label = f"{entry.request.name!r} ({entry.request.expression})"
    _require_declared_corrections(compiled, context.corrections)
    fixed = require_real_fixed_parameters(context.preparation.fixed_parameters)
    scheme = scheme_metadata(compiled)
    if scheme is not None:
        # The card's parameters of value 1 are constants of the complex-mass
        # model (MG5_aMC's restriction): no request inputs.
        fixed = {k: v for k, v in fixed.items() if k not in scheme.get("ones", ())}
    veto = particle_veto_codes(compiled, run.generation.particle_veto)
    pdgs = (*entry.process_ir.initial_pdgs, *entry.process_ir.final_pdgs)
    # Under the complex-mass scheme external states are stable (EW-25).
    require_stable_external_states(compiled, pdgs, label)
    require_native_pole_masses(compiled, label)
    external = sorted({abs(int(pdg)) for pdg in pdgs} & set(veto))
    if external:
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            "veto-external-particle",
            f"{label}: generation.particle_veto removes external particles "
            f"(|PDG| {external}); a veto only applies to internal lines and loops",
        )
    born, dag = born_amplitude_orders(compiled, process, limits, veto)
    if not born:
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            "loop-induced",
            f"{label} has no Born amplitude within the coupling-order limits "
            f"{dict(sorted(limits.items()))}",
        )
    require_zero_coloured_widths(label, (dag,), compiled, fixed)
    accuracy = run.color.accuracy
    selection = (
        _v2_selection(backend, compiled, process, label)
        if _leading_colour(run)
        else None
    )
    request = SliceRequest(
        process,
        dict(limits),
        v2_loop_orders(born, context.corrections, limits),
        corrections=context.corrections,
        fixed_parameters=fixed,
        particle_veto=veto,
        allow_anomalous_veto=run.generation.allow_anomalous_veto,
        accuracy=accuracy,
        colour_declaration=context.color_declaration if accuracy != "full" else None,
        scale_parameter=_v2_scale_parameter(compiled, fixed),
        alpha_s_parameter=run.loop.alpha_s_parameter,
        colour_contraction=_V2_CONTRACTION[run.color.contraction],
        compatibility=context.preparation.compatibility,
        selection=selection,
        external_photons=run.loop.external_photons,
        alpha0_conversion=run.loop.alpha0_conversion_parameter,
        alpha0_inverse=run.loop.alpha0_inverse_parameter,
    )
    if set(context.corrections) - {correction_orders(compiled, fixed).strong}:
        # An electroweak correction ([U8], ``ew_policy.require_zero_widths``):
        # the Born's widths here, where the dry run sees them; the slice
        # checks its loop-side trees too.
        require_zero_widths(compiled, request, (dag,), fixed)
    if born_dags is not None:
        born_dags.append(dag)
    return request


def _born_members(backend, context):
    """The slice requests of the expanded processes that have a Born
    amplitude, with ``context`` restricted to them, and the skipped members
    (m-sm PLAN Q2 (a)).

    A concrete process of a multiparticle request (``p p > e+ e-``: its
    member ``g g > e+ e-``) with no Born amplitude within the coupling-order
    limits is loop-induced: it is skipped with a typed record (key
    ``loop-induced``, the dry run's ``skipped_members`` and the artifact's
    ``loop_skipped_members`` extension), and the Born x virtual sum of the
    request is that of its other members (the loop-squared contribution of
    such a member, which MadLoop's multiparticle output holds, comes with
    the loop-induced milestone). A request with no member left, or a single
    concrete process, is refused (``loop-induced``)."""

    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    requests, dags, kept, skipped = [], [], [], []
    for index, entry in enumerate(context.expanded):
        try:
            request = _v2_request(backend, context, index, dags)
        except LoopRequestUnsupported as error:
            if error.key != "loop-induced" or entry.source_expansion_size <= 1:
                raise
            skipped.append(
                {
                    "key": entry.request.name,
                    "process": entry.request.expression,
                    "request": entry.source_request.name,
                    "request_expression": entry.source_request.expression,
                    "refusal": error.key,
                    "reason": str(error),
                }
            )
            continue
        requests.append(request)
        kept.append(index)
    sources = {context.expanded[index].source_request.name for index in kept}
    empty = sorted(
        {
            (record["request"], record["request_expression"])
            for record in skipped
            if record["request"] not in sources
        }
    )
    if empty:
        name, expression = empty[0]
        raise LoopRequestUnsupported(
            "loop-induced",
            f"no concrete process of {name!r} ({expression}) has a Born "
            "amplitude within the coupling-order limits: "
            + ", ".join(
                record["process"] for record in skipped if record["request"] == name
            ),
        )
    restricted = replace(
        context,
        expanded=tuple(context.expanded[index] for index in kept),
        limits=tuple(context.limits[index] for index in kept),
    )
    return restricted, tuple(requests), tuple(dags), tuple(skipped)


def _plan_policy_checks(compiled, request, model, dag):
    """The dry run's checks of an electroweak request that need only the
    model and the Born DAG (``LoopSlice.generate`` makes them again on the
    Born and the loop-side trees): the external-photon convention
    (``external-photon-scheme``). The width policy on the Born currents
    (``ew_policy.require_zero_widths``, m-sm MS-CMS's) is ``_v2_request``'s,
    which the dry run calls first. ``model`` is the request's
    ``prepare_request_model``."""

    from pyamplicol.loops.v2 import ew_policy
    from pyamplicol.loops.v2.assemble import resolve_request
    from pyamplicol.processes.model import build_model_process_ir

    resolved = resolve_request(compiled, request)
    species = model.prepared.structure.perturbative_species_orders
    if not ew_policy.electroweak(compiled, resolved, species):
        return
    process = build_model_process_ir(
        resolved.process, compiled.ir, color_accuracy="full"
    )
    fixed = model.fixed_parameters
    ew_policy.external_photon_scheme(compiled, resolved, process, fixed, species)


def _plan_v2(backend, processes, blockers, *, model=None):
    """The v2 dry run: per process the slice request (``_v2_request``) and
    the model preparation of the slice (``prepare_request_model``), whose
    refusals are reported. Loop admission (counterterm sites, fermion signs
    of loop vertices, the loop-side tree currents) and the native export of
    the point inputs are checked only by the generation
    (``loop_admission_checked`` and ``native_export_checked`` are False;
    ``native_export_refusals`` lists the features the export refuses,
    ``native_trees.NATIVE_EXPORT_REFUSALS``: such a process fails generation,
    a stored plan v4 process never regenerates at load;
    ``loop_admission_refusals`` the structures the generation refuses,
    ``refusals.LOOP_ADMISSION_REFUSALS``, with their remedies;
    ``process_request_refusals`` the process requests refused for tree and
    loop alike, ``processes.refusals.PROCESS_REQUEST_REFUSALS``)."""
    from pyamplicol.loops.v2.artifact import request_record
    from pyamplicol.loops.v2.assemble import prepare_request_model
    from pyamplicol.loops.v2.native_trees import NATIVE_EXPORT_REFUSALS
    from pyamplicol.loops.v2.refusals import (
        LOOP_ADMISSION_REFUSALS,
        LOOP_CONFIGURATION_REFUSALS,
        LOOP_INTERNAL_REFUSALS,
        keyed_error,
    )
    from pyamplicol.processes.refusals import PROCESS_REQUEST_REFUSALS

    blockers = blockers + _v2_blockers(backend)
    context, coverage, skipped = None, [], ()
    if not blockers:
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
            context, requests, dags, skipped = _born_members(backend, context)
            prepared = {}
            for entry, request, dag in zip(
                context.expanded, requests, dags, strict=True
            ):
                recipe = request_record(request)
                del recipe["process"]
                key = json.dumps(recipe, sort_keys=True)
                if key not in prepared:
                    prepared[key] = prepare_request_model(
                        context.resolved.compiled, request
                    )
                _plan_policy_checks(
                    context.resolved.compiled, request, prepared[key], dag
                )
                coverage.append(
                    {
                        "key": entry.request.name,
                        "process": entry.request.expression,
                        "engine": "v2",
                        "born_orders": dict(request.born_orders),
                        "loop_orders": _loop_orders_record(request),
                        "corrections": list(request.corrections),
                        "color_accuracy": request.accuracy,
                        "scale_parameter": request.scale_parameter,
                        # The shared recursion enumerates no loop diagram.
                        "loop_graph_count": 0,
                        "numerator_compilation_deferred": True,
                        "loop_admission_checked": False,
                        "native_export_checked": False,
                    }
                )
        except (ValueError, TypeError, RuntimeError, ImportError, PyAmpliColError) as e:
            # A listed key first (``refusals.keyed_error``).
            blockers = (str(keyed_error(e)),)
    # Plain texts (a ``ListedBlocker`` is a ``str`` that also carries its key).
    result = _plan_result(
        backend,
        processes,
        tuple(str(blocker) for blocker in blockers),
        context=context,
        coverage=coverage,
    )
    estimated = dict(result.estimated_coverage)
    estimated.update(
        engine="v2",
        arithmetic=("double", "double-double", "arbitrary"),
        resolved=True,
        correlations=True,
        exact_flavor_grouping=backend._run_config.process.flavor_blind,
        api_bundle=False,
        runtime_capability=LOOP_V2_RUNTIME_CAPABILITY,
        native_export_refusals=tuple(
            f"{refusal.key}: {refusal.feature}" for refusal in NATIVE_EXPORT_REFUSALS
        ),
        loop_admission_refusals=tuple(
            f"{refusal.key}: {refusal.feature}; instead: {refusal.remedy}"
            for refusal in LOOP_ADMISSION_REFUSALS
        ),
        loop_configuration_refusals=tuple(
            f"{refusal.key}: {refusal.feature}; instead: {refusal.remedy}"
            for refusal in LOOP_CONFIGURATION_REFUSALS
        ),
        loop_internal_refusals=tuple(
            f"{refusal.key}: {refusal.feature}" for refusal in LOOP_INTERNAL_REFUSALS
        ),
        # Members of multiparticle requests without a Born (m-sm Q2 (a)).
        skipped_members=tuple(skipped),
        process_request_refusals=tuple(
            f"{refusal.key}: {refusal.feature}; instead: {refusal.remedy}"
            for refusal in PROCESS_REQUEST_REFUSALS
        ),
    )
    return replace(result, estimated_coverage=estimated)


# The artifact extension of the skipped members of multiparticle requests
# (``_born_members``, m-sm PLAN Q2 (a)).
SKIPPED_MEMBERS_EXTENSION = "loop_skipped_members"


def _skipped_members_hook(skipped):
    """The manifest extension recording the skipped members (no files)."""

    def hook(_builder):
        return {
            SKIPPED_MEMBERS_EXTENSION: {
                "schema_version": 1,
                "members": [dict(record) for record in skipped],
            }
        }

    return hook


def _v2_member(generated, request, expression):
    """An identity-port flavour member of an already generated process
    (``observables.admit_member``: the exact species-substitution proof on
    the representative's retained source plan), or None. The member's request
    must equal the representative's but for the process."""
    from pyamplicol.loops.v2.artifact import request_record
    from pyamplicol.loops.v2.observables import admit_member

    def recipe(record):
        return {
            key: value
            for key, value in record.items()
            if key not in ("process", "correlators")
        }

    wanted = recipe(request_record(request))
    for name, process in generated.items():
        if recipe(process.request) != wanted:
            continue
        try:
            member = admit_member(process.slice, expression)
        except (ValueError, NotImplementedError):
            continue
        if member.alias or not member.species:
            # Permutation aliases and identical requests stay independent.
            continue
        return name, member
    return None


def _generate_v2(backend, processes, output, *, model, mode, correlators):
    """Generate every process through v2 and publish plan v4 atomically.

    With ``process.flavor_blind`` an identity-port flavour member of an
    earlier process (equal request but for its flavours, an exactly proved
    species substitution) is served by that representative and listed in
    ``process-groups.json``; every other process is generated on its own.
    """
    from pyamplicol.artifacts.process_groups import (
        grouping_process,
        initial_process_groups,
    )
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.artifact import generate_process, write_plan

    from .process_grouping import GroupingBasis, GroupMember, LegMapping

    blockers = _blocked(backend, processes)
    if blockers:
        # The first listed configuration refusal, typed (its key); the
        # message names every one.
        first = blockers[0]
        if len(blockers) == 1:
            raise first.error()
        from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

        raise LoopRequestUnsupported(
            first.key,
            first.message + "; also: " + "; ".join(blockers[1:]),
        )
    reporter = GenerationPhaseReporter(backend._progress)
    reporter.start_generation(
        total=3, details={"execution_mode": "loop_recurrence", "engine": "v2"}
    )
    started = time.perf_counter()
    grouping = backend._run_config.process.flavor_blind
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
            # Members with a Born (m-sm Q2 (a)); the others are recorded.
            context, requests, _dags, skipped = _born_members(backend, context)
        native = load_loop_native()
        compiled = context.resolved.compiled
        validation = backend._generation_config.validation
        with TemporaryDirectory(prefix="pyamplicol-loop-v2-") as temporary:
            artifacts, generated, members = [], {}, {}
            with reporter.phase(
                "loop-compile",
                "Generating v2 loop families and plan v4",
                total=len(context.expanded),
            ) as phase:
                for index, entry in enumerate(context.expanded):
                    request = requests[index]
                    name = entry.request.name
                    found = (
                        _v2_member(generated, request, entry.request.expression)
                        if grouping
                        else None
                    )
                    if found is not None:
                        members.setdefault(found[0], []).append((index, found[1]))
                        phase.update(index + 1, message=f"{name} (flavour member)")
                        continue
                    process = generate_process(compiled, request, native, correlators)
                    generated[name] = process
                    path = Path(temporary) / f"{name}.pacbin"
                    write_plan(
                        process.plan_json,
                        path,
                        native=process.native,
                        scaleless_json=process.scaleless_plan_json,
                    )
                    legs = process.state_legs()
                    # WP3.2 EWI-7: the conventions of an electroweak virtual
                    # (None otherwise).
                    conventions = process.conventions()
                    artifacts.append(
                        LoopV2ProcessArtifact(
                            process_id=name,
                            expression=entry.request.expression,
                            external_pdgs=tuple(int(leg["pdg"]) for leg in legs),
                            loop_runtime_path=path,
                            metadata=LoopV2ExecutionMetadata(
                                scheme=context.declaration.scheme,
                                born_orders=process.born_orders(),
                                sectors=process.sectors(),
                                families=len(process.exported.families),
                                buckets=tuple(map(tuple, process.buckets())),
                                color_accuracy=request.accuracy,
                                color_degrees=process.colour_degrees(),
                                color_contraction=request.colour_contraction,
                                color_correlations=process.color_correlations(),
                                spin_correlations=process.spin_correlations(),
                            ),
                            validation_point=build_process_validation_point(
                                entry.process_ir,
                                context.resolved.model,
                                process_id=name,
                                seed=validation.seed + index,
                            ),
                            generation_filters={
                                "loop_engine": "v2",
                                "loop_coupling_orders": context.corrections,
                                "born_orders": dict(request.born_orders),
                                "loop_orders": _loop_orders_record(request),
                                "particle_veto": list(
                                    backend._run_config.generation.particle_veto
                                ),
                                # [U2]: the override of an anomalous veto and
                                # the CT rules it lowered (rule -> cells),
                                # recorded when set.
                                **(
                                    {
                                        "allow_anomalous_veto": True,
                                        "counterterms_anomalous_veto": (
                                            process.anomalous_veto_counterterms()
                                        ),
                                    }
                                    if request.allow_anomalous_veto
                                    else {}
                                ),
                                # REQUIREMENTS_ADDENDUM J: rule -> reason.
                                "counterterms_inert": process.inert_counterterms(),
                                # The request's gauge (WP3.2 EWI-4b), recorded
                                # off the default unitary gauge only.
                                **(
                                    {"gauge": gauge}
                                    if (gauge := process.slice.compiled.ir.gauge)
                                    != "unitary"
                                    else {}
                                ),
                                # WP3.2 EWI-7: the conventions of an
                                # electroweak virtual (absent otherwise).
                                **(
                                    {"conventions": conventions}
                                    if conventions is not None
                                    else {}
                                ),
                                # RUST-MIGRATION rule 2: stage -> listed feature
                                # (absent when every stage was native).
                                **(
                                    {"native_fallbacks": fallbacks}
                                    if (fallbacks := process.native_fallbacks())
                                    else {}
                                ),
                                # G2-M3: masses whose binary64 lanes take the
                                # exact engine (absent when none is that light).
                                **(
                                    {"binary64_exact_engine": light}
                                    if (light := process.binary64_exact_engine())
                                    else {}
                                ),
                                "model_declaration": context.declaration.to_dict(),
                                **(
                                    {"correlators": correlators.to_json_dict()}
                                    if correlators is not None
                                    else {}
                                ),
                            },
                            color_accuracy=request.accuracy,
                            request=dict(process.request),
                            plan_sha256=process.plan_sha256,
                            scaleless_plan_sha256=process.scaleless_plan_sha256,
                            state_legs=tuple(legs),
                            helicities=tuple(map(tuple, process.helicities())),
                            point_inputs=process.point_inputs,
                        )
                    )
                    phase.update(index + 1, message=name)
            process_groups = None
            if grouping:
                artifacts = [
                    replace(
                        artifact,
                        generation_filters={
                            **artifact.generation_filters,
                            "flavor_grouping": _flavor_grouping_outcome(
                                backend,
                                correlators,
                                reused=bool(members.get(artifact.process_id)),
                            ),
                        },
                    )
                    for artifact in artifacts
                ]
                process_groups = initial_process_groups(
                    (
                        (
                            artifact.process_id,
                            artifact.expression,
                            artifact.external_pdgs,
                        )
                        for artifact in artifacts
                    ),
                    {
                        representative: [
                            GroupMember.identity(
                                replace(
                                    grouping_process(
                                        context.expanded[index].request.name,
                                        context.expanded[index].request.expression,
                                        (
                                            *context.expanded[
                                                index
                                            ].process_ir.initial_pdgs,
                                            *context.expanded[
                                                index
                                            ].process_ir.final_pdgs,
                                        ),
                                    ),
                                    generated=False,
                                )
                            ).related(
                                LegMapping(
                                    tuple(range(len(member.legs))),
                                    False,
                                    "exact-loop-species",
                                ),
                                factor=1,
                                basis=GroupingBasis.EXACT_LOOP_FLAVOR,
                            )
                            for index, member in admitted
                        ]
                        for representative, admitted in members.items()
                    },
                )
            validate = validation.post_build_validation
            with reporter.phase(
                "artifact-writing",
                "Writing and validating virtual artifact"
                if validate
                else "Writing virtual artifact",
                total=1,
            ) as phase:
                result = backend._write_artifact(
                    Path(output),
                    mode=mode,
                    source=context.resolved.source,
                    compiled_model=compiled,
                    configuration=backend._configuration,
                    processes=artifacts,
                    process_groups=process_groups,
                    timings=reporter.timings,
                    api_bundle_hook=None,
                    **(
                        {"payload_hook": _skipped_members_hook(skipped)}
                        if skipped
                        else {}
                    ),
                    staged_validation=(
                        lambda staged: _validate_v2_artifact(
                            backend, staged, context, generated, members
                        )
                    )
                    if validate
                    else None,
                )
                phase.update(1, message=str(result.output))
        reporter.timings["total"] = time.perf_counter() - started
    except BaseException as error:
        reporter.finish_generation(success=False, message=str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit, PyAmpliColError)):
            raise
        from pyamplicol.loops.v2.refusals import keyed_error

        # An engine error class with a listed key (``refusals``: internal
        # and admission keys, native export features) keeps it, typed.
        keyed = keyed_error(error)
        if isinstance(keyed, PyAmpliColError):
            raise keyed from error
        raise GenerationError(str(error)) from error
    reporter.finish_generation()
    return GenerationResult(
        output=result.output,
        processes=ProcessSet(tuple(e.request for e in context.expanded)),
        mode=mode,
        files=result.files,
    )


def _validate_v2_artifact(backend, output, context, generated, members):
    """Load every staged process (and flavour member) through the public
    runtime and compare its native evaluation, inclusive and for every
    declared colour history, with the Python reference executor of the
    generated slice on the validation points, in binary64; a point whose
    binary64 pair misses the tolerance is re-validated in both carriers
    against the reference at 32 digits (``_revalidate_v2_point``)."""
    from pyamplicol.api import Runtime
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.loops.v2.observables import MemberObservables

    manifest = load_manifest(output)
    validation = backend._generation_config.validation
    indices = {entry.request.name: i for i, entry in enumerate(context.expanded)}
    served = [(name, name, None) for name in generated]
    served += [
        (context.expanded[index].request.name, representative, member)
        for representative, admitted in members.items()
        for index, member in admitted
    ]
    for name, representative, member in served:
        index = indices[name]
        entry = context.expanded[index]
        runtime = Runtime.load(output, process=name)
        if (
            runtime.execution_mode != "loop_recurrence"
            or runtime.artifact_id != manifest.artifact_id
            or runtime.physics.process_id != name
            or runtime.representative_process_key != representative
        ):
            raise GenerationError("v2 validation loaded incompatible artifact metadata")
        if not validation.enabled:
            continue
        points, scales = _v2_validation_points(context, entry, validation, index)
        observables = generated[representative].observables
        _validate_v2_points(
            name,
            runtime,
            observables if member is None else MemberObservables(observables, member),
            tuple(history.id for history in observables.histories),
            points,
            scales,
            validation.relative_tolerance,
        )


def _v2_outputs(runtime, points, scales, histories, **options):
    """A runtime's native inclusive (``"born"``) and declared-colour-history
    evaluations of a point batch, by label."""
    from pyamplicol.correlators import CorrelatedRequest

    outputs = {"born": runtime.evaluate(points, mu_r2=scales, **options)}
    if histories:
        outputs |= runtime.evaluate_correlated_many(
            points,
            {history: CorrelatedRequest(history, {}) for history in histories},
            mu_r2=scales,
            **options,
        )
    return outputs


def _validate_v2_points(name, runtime, runner, histories, points, scales, tolerance):
    """Binary64 native outputs of the validation points against the reference
    executor ``runner`` in binary64, at ``tolerance``; a point whose binary64
    pair misses it (``_compare_v2`` refuses) is re-validated alone by
    ``_revalidate_v2_point``, which refuses or records it."""

    outputs = _v2_outputs(runtime, points, scales, histories)
    for point_index, (point, scale) in enumerate(zip(points, scales, strict=True)):
        references = runner.evaluate(point, scale, correlations=("born", *histories))
        try:
            for label, values in outputs.items():
                _compare_v2(
                    name, label, values, point_index, references[label], tolerance
                )
        except GenerationError as miss:
            _revalidate_v2_point(
                name,
                runtime,
                runner,
                histories,
                outputs,
                point_index,
                point,
                scale,
                tolerance,
                miss,
            )


def _revalidate_v2_point(
    name,
    runtime,
    runner,
    histories,
    binary64,
    point_index,
    point,
    scale,
    tolerance,
    miss,
) -> None:
    """One validation point whose binary64 pair ``miss``ed the tolerance,
    checked against the reference executor at ``DOUBLE_DOUBLE_DIGITS``
    decimal digits, every output of the point (Born and every declared colour
    history), in both carriers:

    - native double-double at the same ``tolerance``;
    - the point's binary64 outputs (``binary64``, by label, as validated):
      Borns at ``tolerance``, virtuals at max(``tolerance``, E), E the
      point's binary64 stability estimate (the recommended checks: one
      generic exact rotation and the rescaled reduction check), refused when
      absent or above 1e-9 (the
      user's O1 rule; above the tolerance where that is looser). A binary64
      virtual off by more than the tolerance is therefore accepted only where
      binary64 itself estimates that it cannot resolve the tolerance.

    The binary64 pair is two routes (native and Python dressing), each
    rounded its own way: at d d~ > g g NLC the casimir virtual's estimate is
    1.5e-11, the native value 2.7e-12 of its sector off the 32-digit value,
    and the pair 1.3e-12 apart. A defect in either carrier is refused, the
    binary64 refusal chained; an accepted point is logged with its record.
    """
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.assemble import DOUBLE_DOUBLE_DIGITS, Precision
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    try:
        native = _v2_outputs(
            runtime, [point], [scale], histories, arithmetic="double-double"
        )
        checked = _v2_outputs(
            runtime,
            [point],
            [scale],
            histories,
            stability_checks=VirtualStabilityChecks(
                RECOMMENDED_CHECKS.rotations, True, rescaled=True
            ),
        )
        references = runner.evaluate(
            point,
            scale,
            correlations=("born", *histories),
            precision=Precision.arbitrary(DOUBLE_DOUBLE_DIGITS),
        )
    except Exception as error:
        raise GenerationError(
            f"{miss}; its re-validation at {DOUBLE_DOUBLE_DIGITS} digits could "
            f"not run: {error}"
        ) from miss
    record = []
    for label, values in native.items():
        try:
            double_double = _compare_v2(
                name, label, values, 0, references[label], tolerance
            )
        except GenerationError as error:
            raise GenerationError(
                f"{error} in double-double, after binary64: {miss}"
            ) from miss
        stability = checked[label].stability
        estimate = None if stability is None else stability[0]
        if estimate is None or not estimate.relative <= max(
            tolerance, _BINARY64_ESTIMATE_CAP
        ):
            raise GenerationError(
                f"{miss}; the binary64 virtual of {name!r} ({label}) has no "
                "admissible stability estimate ("
                f"{None if estimate is None else estimate.relative!r}, at most "
                f"{_BINARY64_ESTIMATE_CAP:g})"
            ) from miss
        try:
            deviation = _compare_v2(
                name,
                label,
                binary64[label],
                point_index,
                references[label],
                tolerance,
                virtual_tolerance=max(tolerance, estimate.relative),
            )
        except GenerationError as error:
            raise GenerationError(
                f"{error} in binary64 against {DOUBLE_DOUBLE_DIGITS} digits, "
                f"beyond its estimate {estimate.relative:.3g}, after: {miss}"
            ) from miss
        record.append(
            f"{label}: binary64 {deviation:.2g} (estimate {estimate.relative:.2g}),"
            f" double-double {double_double:.2g}"
        )
    _LOGGER.warning(
        "v2 validation of %r, point %d: binary64 missed the tolerance %g (%s); "
        "re-validated against the reference at %d digits, deviations relative "
        "to each sector's largest coefficient: %s",
        name,
        point_index,
        tolerance,
        miss,
        DOUBLE_DOUBLE_DIGITS,
        "; ".join(record),
    )


def _compare_v2(
    name, label, result, point_index, reference, tolerance, virtual_tolerance=None
):
    """One native output at ``point_index`` against a reference-executor
    result: every Born order within ``tolerance`` of its own value, every
    virtual Laurent coefficient within ``virtual_tolerance`` (default
    ``tolerance``) of its sector's largest one. Returns the largest virtual
    deviation relative to its sector's largest coefficient."""

    bound = tolerance if virtual_tolerance is None else virtual_tolerance
    born = {canonical_order(k): v for k, v in reference.born.items()}
    for order, value in zip(
        result.born_orders, result.born_values[point_index], strict=True
    ):
        target = born.get(order, 0.0)
        if not cmath.isclose(
            complex(value), complex(target), rel_tol=tolerance, abs_tol=0.0
        ):
            raise GenerationError(
                f"saved v2 Born of {name!r} ({label}) differs from the reference "
                f"executor at order {order} ({value!r} != {target!r})"
            )
    virtual = {
        canonical_order(k): v for k, v in reference.components["virtual"].items()
    }
    largest = 0.0
    for sector, laurent in zip(result.sectors, result.values[point_index], strict=True):
        target = virtual[sector.order].coefficients
        size = max(abs(complex(x)) for x in target)
        for k, (value, expected) in enumerate(zip(laurent, target, strict=True)):
            difference = abs(complex(value) - complex(expected))
            if difference > bound * size:
                raise GenerationError(
                    f"saved v2 virtual of {name!r} ({label}) differs from the "
                    f"reference executor (sector {sector.order}, coefficient {k}: "
                    f"{value!r} != {expected!r})"
                )
            if difference:
                largest = max(largest, difference / size)
    return largest


def _v2_validation_points(context, entry, validation, index):
    from fractions import Fraction

    points, scales = [], []
    for sample in range(validation.samples):
        record = build_process_validation_point(
            entry.process_ir,
            context.resolved.model,
            process_id=entry.request.name,
            seed=validation.seed + index + sample,
        )
        if not record.available:
            raise GenerationError(
                f"virtual validation momenta unavailable: {record.error}"
            )
        point = [[repr(float(x)) for x in vector] for vector in record.four_vectors]
        initial = len(entry.process_ir.initial_pdgs)
        total = [
            sum(Fraction(vector[mu]) for vector in point[:initial]) for mu in range(4)
        ]
        scale = total[0] ** 2 - sum(component**2 for component in total[1:])
        if scale <= 0:
            raise GenerationError("virtual validation requires a positive invariant")
        points.append(point)
        scales.append(str(Decimal(scale.numerator) / Decimal(scale.denominator)))
    return points, scales
