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
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pyamplicol.api.errors import GenerationError, PyAmpliColError
from pyamplicol.api.requests import ProcessSet
from pyamplicol.api.results import GenerationPlan, GenerationResult
from pyamplicol.config import ColorContraction
from pyamplicol.loops.covariant_model import with_source_repairs
from pyamplicol.loops.model_cms import (
    COMPLEX_MASS,
    apply_mass_scheme,
    require_real_fixed_parameters,
)
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
from .loop_artifact import (
    LoopV2ExecutionMetadata,
    LoopV2ProcessArtifact,
    loop_v2_capability,
)
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
    # Permutation aliases are published (``loops.v2.mapping``); a relation
    # found only by numerical comparison is not.
    if run.process.analyze_process_groups != "none":
        reasons.append(
            ListedBlocker(
                "process-group-analysis",
                "public loop artifacts map processes by exact proofs only "
                "(permutation aliases, process.flavor_blind species "
                "substitutions), never by a numerical process-group analysis "
                "(process.analyze_process_groups = "
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
    # ``process.max_coupling_orders`` by the declared names: the bounds of a
    # loop-induced request's loop amplitudes (``loop_induced_orders``).
    explicit: dict[str, int] = field(default_factory=dict)
    # Per entry the loop orders of a loop-induced request that are exact
    # colour zeros (``_loop_induced_request``), None for a Born member.
    colour_zero: tuple[tuple[dict[str, int], ...] | None, ...] = ()


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
        explicit=dict(explicit),
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
        reason = "no exact member was admitted"
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
    ``_member`` / ``_v2_request`` / ``_loop_induced_request`` (undeclared
    corrections, coloured widths in the Born currents of a Born-interfered
    request, an unidentified scale parameter, the loop-induced routing of
    ``_born_members``) and
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


def _raise_blocked(backend, processes) -> None:
    """Raise the first listed configuration refusal of ``_blocked``, typed
    (its key); the message names every one."""

    blockers = _blocked(backend, processes)
    if not blockers:
        return
    first = blockers[0]
    if len(blockers) == 1:
        raise first.error()
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    raise LoopRequestUnsupported(
        first.key,
        first.message + "; also: " + "; ".join(blockers[1:]),
    )


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


@dataclass(frozen=True)
class _Member:
    """The request-independent facts of one expanded process (``_member``):
    its process, label, coupling-order limits, fixed parameters, vetoed
    |PDG| codes, Born amplitude orders within the limits and the Born DAG
    they were read from (empty orders: no Born amplitude, loop-induced)."""

    process: str
    label: str
    limits: dict[str, int]
    fixed: dict[str, str]
    veto: tuple[int, ...]
    born: tuple
    dag: Any


def _member(backend, context, index) -> _Member:
    """The checks every expanded process takes before it is routed (declared
    corrections, stable external states under the complex-mass scheme,
    native pole masses, no vetoed external particle) and its Born amplitude
    orders within its coupling-order limits (``artifact.born_amplitude_orders``):
    a process is loop-induced iff no Born DAG root exists within them
    (m-lit PLAN §1.2; never a numerical zero)."""

    from pyamplicol.loops.model_cms import (
        require_native_pole_masses,
        require_stable_external_states,
        scheme_metadata,
    )
    from pyamplicol.loops.v2.artifact import born_amplitude_orders

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
    # Under the complex-mass scheme external states are stable (EW-25), for
    # Born-interfered and loop-induced requests alike (m-lit Q7).
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
    return _Member(process, label, dict(limits), fixed, veto, born, dag)


def _no_born(member: _Member):
    """The typed ``loop-induced`` record (or refusal) of a Born-less process
    a request does not serve."""

    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    return LoopRequestUnsupported(
        "loop-induced",
        f"{member.label} has no Born amplitude within the coupling-order limits "
        f"{dict(sorted(member.limits.items()))}",
    )


def _v2_request(backend, context, index, born_dags=None, member=None):
    """The slice request of one expanded process with a Born amplitude
    (``loops.v2.assemble``), checked before any loop generation; its Born DAG
    is appended to ``born_dags`` when a list is given (the dry run's policy
    checks read it). ``member`` is the process's ``_member`` (made here when
    None).

    The Born bounds are the process's coupling-order limits. The loop orders
    follow its Born amplitude orders (``v2_loop_orders``), read from the Born
    DAG (``artifact.born_amplitude_orders``), which must exist (else the
    typed ``loop-induced`` refusal, which ``_born_members`` routes) and
    propagate no coloured particle with a width
    (``artifact.require_zero_coloured_widths``; the slice re-checks its
    loop-side tree currents after generation). The renormalisation-scale
    parameter is resolved here (``_v2_scale_parameter``)."""
    from pyamplicol.loops.v2.artifact import require_zero_coloured_widths
    from pyamplicol.loops.v2.ew_policy import correction_orders, require_zero_widths

    run = backend._run_config
    compiled = context.resolved.compiled
    if member is None:
        member = _member(backend, context, index)
    if not member.born:
        raise _no_born(member)
    limits, label, fixed, dag = member.limits, member.label, member.fixed, member.dag
    require_zero_coloured_widths(label, (dag,), compiled, fixed)
    accuracy = run.color.accuracy
    selection = (
        _v2_selection(backend, compiled, member.process, label)
        if _leading_colour(run)
        else None
    )
    request = _slice_request(
        backend,
        context,
        member,
        v2_loop_orders(member.born, context.corrections, limits),
        accuracy=accuracy,
        colour_declaration=context.color_declaration if accuracy != "full" else None,
        selection=selection,
    )
    if set(context.corrections) - {correction_orders(compiled, fixed).strong}:
        # An electroweak correction ([U8], ``ew_policy.require_zero_widths``):
        # the Born's widths here, where the dry run sees them; the slice
        # checks its loop-side trees too.
        require_zero_widths(compiled, request, (dag,), fixed)
    if born_dags is not None:
        born_dags.append(dag)
    return request


def _slice_request(backend, context, member, loop_orders, **fields):
    """The slice request of ``member`` (``_member``) at ``loop_orders`` with
    the run's settings, its coupling-order limits as Born bounds;
    ``fields`` sets the others (accuracy, colour declaration, selection,
    contribution). Shared by ``_v2_request`` and ``_loop_induced_request``."""

    from pyamplicol.loops.v2.assemble import SliceRequest

    run = backend._run_config
    return SliceRequest(
        member.process,
        dict(member.limits),
        loop_orders,
        corrections=context.corrections,
        fixed_parameters=member.fixed,
        particle_veto=member.veto,
        allow_anomalous_veto=run.generation.allow_anomalous_veto,
        scale_parameter=_v2_scale_parameter(context.resolved.compiled, member.fixed),
        alpha_s_parameter=run.loop.alpha_s_parameter,
        colour_contraction=_V2_CONTRACTION[run.color.contraction],
        compatibility=context.preparation.compatibility,
        external_photons=run.loop.external_photons,
        alpha0_conversion=run.loop.alpha0_conversion_parameter,
        alpha0_inverse=run.loop.alpha0_inverse_parameter,
        reduction=str(run.loop.reduction),
        **fields,
    )


def _order_hierarchy(context) -> dict[str, int]:
    """The model's coupling-order hierarchy by declared name (at least 1):
    the weights of the tree service's minimal-order policy
    (``coupling_order_hierarchies``, read by
    ``dag_algorithms.infer_minimal_coupling_order_limits``)."""

    compiled = context.resolved.compiled
    return _order_names(compiled).mapping(
        context.resolved.model.coupling_order_hierarchies()
    )


def loop_induced_orders(
    reachable: Iterable[Sequence[tuple[str, int]]],
    *,
    policy: str,
    hierarchy: Mapping[str, int] | None = None,
    names: Iterable[str] = (),
) -> dict[str, int] | tuple[dict[str, int], ...]:
    """The exact loop-amplitude orders of a loop-induced request (m-lit PLAN
    §1.3, Q4 (a); the ``v2_loop_orders`` of a process without a Born) from
    the orders its field-level loop family recursion reaches within the
    request's bounds (``reachable``; ``assemble.reachable_loop_orders``).

    ``explicit``: every reachable order (``process.max_coupling_orders``
    bounds the amplitude orders). ``minimal``: the lowest reachable ones by
    the model's coupling hierarchy, i.e. every reachable order of the least
    weighted degree sum(hierarchy * power) (as MG5_aMC, which bounds the
    WEIGHTED order by that of its lowest loop diagram,
    ``loop_diagram_generation.py``). Squared orders are the loop x loop
    pairs l + l'. A mapping for one order, else
    a sequence by degree; every mapping names ``names`` and every reached
    coupling (zero powers kept). Refused (``ValueError``) without a
    reachable order: the caller's ``no-loop-amplitude``."""

    found = sorted(
        {tuple(sorted((str(n), int(p)) for n, p in o if p)) for o in reachable}
    )
    if not found:
        raise ValueError("a loop-induced request needs a reachable loop amplitude")
    weights = dict(hierarchy or {})

    def degree(orders) -> int:
        return sum(max(1, int(weights.get(name, 1))) * power for name, power in orders)

    if str(policy) == "minimal":
        lowest = min(map(degree, found))
        found = [orders for orders in found if degree(orders) == lowest]
    elif str(policy) != "explicit":
        raise ValueError(f"unknown coupling-order policy {policy!r}")
    keys = sorted({*map(str, names), *(name for orders in found for name, _ in orders)})
    loops = []
    for orders in sorted(found, key=lambda orders: (degree(orders), orders)):
        loop = dict.fromkeys(keys, 0)
        loop.update(orders)
        loops.append(loop)
    return loops[0] if len(loops) == 1 else tuple(loops)


def _loop_induced_request(backend, context, index, member, memo):
    """The slice request of one expanded process without a Born amplitude,
    served loop-induced (|A_loop|^2; m-lit LIT-REQ), and its loop orders
    that are exact colour zeros. ``memo`` (the generation's
    ``preparation_cache.PreparationMemo``) shares the model preparation of
    the order inference by recipe (every request's refusals run for it)
    and its lowered declarations with every other preparation.

    Full colour only (``loop-induced-colour-accuracy``, Q8). The loop orders
    are inferred (``loop_induced_orders``) from the field-level loop family
    recursion within ``process.max_coupling_orders``
    (``assemble.reachable_loop_orders``): the orders with a loop amplitude
    and those every loop of which has a single coloured attachment (an
    exact colour zero, Tr T^a = 0: ``h > g a``; the request is one when all
    its orders are); none reachable is the typed ``no-loop-amplitude``.
    Widths (Q7): fixed widths are admitted in every tree propagator (no
    coloured-width refusal: with no Born there is no Catani mismatch, and
    the UV counterterm vertices multiply the same widthful tree
    propagators; the runtime finiteness diagnostic checks it at production
    widths), loop propagators stay widthless; under the complex-mass scheme
    masses are complex everywhere and unstable external states were refused
    by ``_member``. The electroweak width guard of a Born-interfered
    request (``ew_policy.require_zero_widths``, ``electroweak-width``) does
    not apply (m-lit LIT-EW): an electroweak loop-induced request takes the
    same Q7 policy, which ``LoopSlice.generate`` records
    (``ew_policy.tree_width_policy``)."""

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2 import generator_input
    from pyamplicol.loops.v2.assemble import (
        prepare_request_model,
        reachable_loop_orders,
    )
    from pyamplicol.loops.v2.families import add_orders
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    run = backend._run_config
    compiled = context.resolved.compiled
    accuracy = str(run.color.accuracy)
    if accuracy != "full":
        raise LoopRequestUnsupported(
            "loop-induced-colour-accuracy",
            f"{member.label} has no Born amplitude within the coupling-order "
            f"limits {dict(sorted(member.limits.items()))} and requests "
            f"color.accuracy = {accuracy!r}",
        )
    bounds = dict(context.explicit)
    # The request before its loop orders are known: bounded by the explicit
    # limits only, its preparation keeps every vertex the bounds admit.
    base = _slice_request(
        backend,
        context,
        member,
        bounds,
        accuracy="full",
        contribution="loop-induced",
    )
    model = prepare_request_model(compiled, base, memo=memo)
    native = load_loop_native() if "families" in generator_input.NATIVE_STAGES else None
    reachable = reachable_loop_orders(
        compiled, base, bounds, model=model, native=native
    )
    if not reachable.orders:
        raise LoopRequestUnsupported(
            "no-loop-amplitude",
            f"{member.label} has no Born amplitude within the coupling-order "
            f"limits {dict(sorted(member.limits.items()))} and no one-loop "
            f"amplitude for the corrections {list(context.corrections)} within "
            f"{dict(sorted(bounds.items())) or 'the derived order bounds'}",
        )
    orders = loop_induced_orders(
        reachable.orders,
        policy=str(run.process.coupling_order_policy),
        hierarchy=_order_hierarchy(context),
        names=(*_order_names(compiled).declared, *bounds, *context.corrections),
    )
    request = replace(base, loop_orders=orders)
    zero = tuple(
        dict(loop)
        for loop in request.exact_loop_orders
        if add_orders(tuple(loop.items())) in reachable.colour_zero
    )
    return request, zero


def _skip_record(entry, error) -> dict[str, str]:
    return {
        "key": entry.request.name,
        "process": entry.request.expression,
        "request": entry.source_request.name,
        "request_expression": entry.source_request.expression,
        "refusal": error.key,
        "reason": str(error),
    }


def _born_members(backend, context, memo=None):
    """The slice requests of the expanded processes a request serves, with
    ``context`` restricted to them (and ``colour_zero`` set), their Born
    DAGs (None for a loop-induced one) and the skipped members, routed by
    ``process.loop_induced`` (m-lit PLAN Q2 (b), §1.7; m-sm PLAN Q2 (a)).

    A concrete process with no Born amplitude within its coupling-order
    limits (``_member``) is loop-induced. Per request (a ``ProcessRequest``
    and its concrete processes):

    * ``auto`` (default): a request with a Born member serves its Born
      members; its Born-less members (``p p > e+ e-``: ``g g > e+ e-``) are
      skipped with the typed ``loop-induced`` record (the dry run's
      ``skipped_members``, the artifact's ``loop_skipped_members``). A
      request none of whose members has a Born serves them loop-induced
      (``_loop_induced_request``).
    * ``skip``: Born members only; a request with no Born member, or a
      single Born-less process, is refused (``loop-induced``).
    * ``only``: Born-less members only, loop-induced; Born members are
      skipped with the typed ``born-in-loop-induced-mode`` record, and a
      request without a Born-less member is refused with that key.
    * ``include``: both kinds, as separate contributions (each concrete
      process its own, never summed): Born members as Born-interfered
      virtuals, Born-less members loop-induced (m-lit LIT-API).

    A loop-induced member that reaches no loop amplitude (not even an exact
    colour zero, which is served) is skipped with the typed
    ``no-loop-amplitude`` record (refused for a single process). A
    request left without a member is refused with the key of its records.
    Born members are processed in order exactly as before (``_v2_request``);
    the routing of Born-less members waits for their request's other
    members. ``memo`` (the generation's ``preparation_cache.PreparationMemo``;
    None: one for this call) shares the order inference's model
    preparations."""

    from pyamplicol.loops.preparation_cache import PreparationMemo
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    policy = str(backend._run_config.process.loop_induced)
    served: dict[int, tuple[Any, Any, Any]] = {}
    skipped: dict[int, dict[str, str]] = {}
    deferred: list[tuple[int, _Member]] = []
    with_born: set[str] = set()
    memo = PreparationMemo() if memo is None else memo
    for index, entry in enumerate(context.expanded):
        member = _member(backend, context, index)
        single = entry.source_expansion_size <= 1
        if member.born:
            with_born.add(entry.source_request.name)
            if policy == "only":
                error = LoopRequestUnsupported(
                    "born-in-loop-induced-mode",
                    f"{member.label} has a Born amplitude within the "
                    "coupling-order limits "
                    f"{dict(sorted(member.limits.items()))} (orders "
                    f"{[dict(orders) for orders in member.born]}) and "
                    "process.loop_induced = 'only'",
                )
                if single:
                    raise error
                skipped[index] = _skip_record(entry, error)
                continue
            request = _v2_request(backend, context, index, None, member)
            served[index] = (request, member.dag, None)
        elif single:
            if policy == "skip":
                raise _no_born(member)
            request, zero = _loop_induced_request(backend, context, index, member, memo)
            served[index] = (request, None, zero)
        else:
            deferred.append((index, member))
    for index, member in deferred:
        entry = context.expanded[index]
        if policy == "skip" or (
            policy == "auto" and entry.source_request.name in with_born
        ):
            skipped[index] = _skip_record(entry, _no_born(member))
            continue
        try:
            request, zero = _loop_induced_request(backend, context, index, member, memo)
        except LoopRequestUnsupported as error:
            if error.key != "no-loop-amplitude":
                raise
            skipped[index] = _skip_record(entry, error)
            continue
        served[index] = (request, None, zero)
    kept = sorted(served)
    records = tuple(skipped[index] for index in sorted(skipped))
    sources = {context.expanded[index].source_request.name for index in kept}
    empty = sorted(
        {
            (record["request"], record["request_expression"])
            for record in records
            if record["request"] not in sources
        }
    )
    if empty:
        name, expression = empty[0]
        mine = [record for record in records if record["request"] == name]
        listed = ", ".join(record["process"] for record in mine)
        refusals = {record["refusal"] for record in mine}
        if "no-loop-amplitude" in refusals:
            raise LoopRequestUnsupported(
                "no-loop-amplitude",
                f"no concrete process of {name!r} ({expression}) has a Born or a "
                "loop amplitude within its coupling-order limits and corrections: "
                + listed,
            )
        if refusals == {"born-in-loop-induced-mode"}:
            raise LoopRequestUnsupported(
                "born-in-loop-induced-mode",
                f"every concrete process of {name!r} ({expression}) has a Born "
                "amplitude within its coupling-order limits and "
                "process.loop_induced = 'only': " + listed,
            )
        raise LoopRequestUnsupported(
            "loop-induced",
            f"no concrete process of {name!r} ({expression}) has a Born "
            "amplitude within the coupling-order limits: " + listed,
        )
    restricted = replace(
        context,
        expanded=tuple(context.expanded[index] for index in kept),
        limits=tuple(context.limits[index] for index in kept),
        colour_zero=tuple(served[index][2] for index in kept),
    )
    return (
        restricted,
        tuple(served[index][0] for index in kept),
        tuple(served[index][1] for index in kept),
        records,
    )


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
    loop alike, ``processes.refusals.PROCESS_REQUEST_REFUSALS``). Each
    process names its runtime capability (from its request's contribution:
    v4 Born-interfered, v5 loop-induced); ``runtime_capabilities`` is the
    planned artifact's union, ``runtime_capability`` the one runtime when
    every process shares it (else None). One ``preparation_cache.
    PreparationMemo`` serves the order inference's and the processes'
    model preparations (by recipe; every request's refusals run for it),
    with the persistent store of lowered programs unless it is off
    (``preparation_cache.store_for``): a dry run fills it when the compiled
    model is available (it never compiles one)."""
    from pyamplicol.loops.preparation_cache import PreparationMemo, store_for
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
            memo = PreparationMemo(store_for(backend._run_config.model))
            context, requests, dags, skipped = _born_members(backend, context, memo)
            for entry, request, dag, zero in zip(
                context.expanded, requests, dags, context.colour_zero, strict=True
            ):
                prepared = prepare_request_model(
                    context.resolved.compiled, request, memo=memo
                )
                _plan_policy_checks(context.resolved.compiled, request, prepared, dag)
                record = (
                    _loop_induced_coverage(backend, context, entry, request, zero)
                    if request.contribution == "loop-induced"
                    else {
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
                # The process's runtime: from its request's contribution.
                record["runtime_capability"] = loop_v2_capability(request.contribution)
                coverage.append(record)
            _LOGGER.info("loop preparation: %s", json.dumps(memo.report()))
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
    capabilities = tuple(sorted({str(item["runtime_capability"]) for item in coverage}))
    estimated.update(
        engine="v2",
        arithmetic=("double", "double-double", "arbitrary"),
        resolved=True,
        correlations=True,
        exact_flavor_grouping=backend._run_config.process.flavor_blind,
        api_bundle=False,
        # The runtimes the planned artifact requires (its manifest's union,
        # sorted), and the one runtime when every process shares it (None for
        # a mix of Born-interfered and loop-induced processes, or no process).
        runtime_capabilities=capabilities,
        runtime_capability=capabilities[0] if len(capabilities) == 1 else None,
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


def _loop_induced_sectors(request) -> list[dict[str, object]]:
    """The squared-order sectors of a loop-induced request: loop x loop pairs
    (l, l'), l <= l', of order l + l' (2 Re for l != l'; m-lit PLAN §1.3)."""

    loops = [dict(orders) for orders in request.exact_loop_orders]
    sectors: dict[tuple, dict[str, object]] = {}
    for i, left in enumerate(loops):
        for right in loops[i:]:
            names = sorted({*left, *right})
            order = {name: left.get(name, 0) + right.get(name, 0) for name in names}
            key = tuple(sorted(order.items()))
            sector = sectors.setdefault(key, {"order": order, "pairs": []})
            sector["pairs"].append([left, right])
    return [sectors[key] for key in sorted(sectors)]


def _loop_induced_coverage(
    backend, context, entry, request, colour_zero
) -> dict[str, object]:
    """The dry run's record of a process served loop-induced (|A_loop|^2):
    its kind, the coupling-order limits within which it has no Born, its
    inferred loop orders and loop x loop sectors, those of its loop orders
    that are exact colour zeros (``colour_zero``; every loop has a single
    coloured attachment, Tr T^a = 0) and whether all are (an exact colour
    zero request, ``h > g a``: m-lit PLAN §1.2, Q13), the order policy and
    the width policy (m-lit Q7)."""

    from pyamplicol.loops.v2.ew_policy import COMPLEX_MASS_WIDTHS, FIXED_TREE_WIDTHS

    run = backend._run_config
    scheme = str(context.declaration.scheme.mass_scheme)
    return {
        "key": entry.request.name,
        "process": entry.request.expression,
        "engine": "v2",
        "contribution": "loop-induced",
        "interference": "conj(loop)*loop",
        "born_orders": {},
        "coupling_order_limits": dict(request.born_orders),
        "loop_orders": _loop_orders_record(request),
        "loop_order_policy": str(run.process.coupling_order_policy),
        "sectors": _loop_induced_sectors(request),
        "colour_zero_loop_orders": [dict(orders) for orders in colour_zero],
        "exact_colour_zero": len(colour_zero) == len(request.exact_loop_orders),
        "corrections": list(request.corrections),
        "color_accuracy": request.accuracy,
        "scale_parameter": request.scale_parameter,
        # m-lit Q7: fixed widths in every tree propagator and widthless loop
        # propagators; the complex-mass scheme puts complex masses in tree
        # and loop propagators (unstable external states refused).
        "width_policy": (
            COMPLEX_MASS_WIDTHS if scheme == COMPLEX_MASS else FIXED_TREE_WIDTHS
        ),
        "loop_graph_count": 0,
        "numerator_compilation_deferred": True,
        "loop_admission_checked": False,
        "native_export_checked": False,
    }


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
    """A member of an already generated process (``observables.
    admit_member``: an exact species substitution proved on the
    representative's retained source plan, a permutation within each side,
    or both), or None. The member's request must equal the representative's
    but for the process; a permuted member's leg-keyed request fields must
    be invariant under its permutation (``_permutation_invariant``) and the
    representative must store the image of every declared colour history and
    spin class (``_correlations_closed``): otherwise the member is generated
    on its own and serves every declared correlation."""
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
        if not (member.alias or member.species):
            # An identical request stays independent.
            continue
        if member.alias and not (
            _permutation_invariant(request, member.legs)
            and _correlations_closed(process.observables, member)
        ):
            continue
        return name, member
    return None


def _correlations_closed(observables, member) -> bool:
    """Whether the representative stores the image of every declared colour
    history and spin class read on the member's labels (``loops.v2.mapping``
    ``mapped_histories`` / ``mapped_spin_classes``), so that folding the
    member loses none of its declared correlations."""
    from pyamplicol.loops.v2.mapping import mapped_histories, mapped_spin_classes

    histories = observables.histories
    classes = observables.spin_classes
    return len(mapped_histories(member, histories)) == len(histories) and len(
        mapped_spin_classes(member, classes)
    ) == len(classes)


def _permutation_invariant(request, legs) -> bool:
    """Whether the leg-keyed fields of a slice request read the same legs
    after the member permutation ``legs`` (member leg i -> representative leg
    ``legs[i]``): no generation-time helicity or colour selection, and an
    external-photon tagging by mode or by a leg set the permutation maps
    onto itself."""

    if request.selection is not None:
        return False
    tagged = request.external_photons
    return isinstance(tagged, str) or {legs[leg - 1] + 1 for leg in tagged} == set(
        tagged
    )


def _generate_v2(backend, processes, output, *, model, mode, correlators):
    """Generate every process through v2 and publish plan v4 atomically.

    With ``process.flavor_blind`` a member of an earlier process (equal
    request but for its process: an exactly proved species substitution, a
    permutation within each side, or both; ``_v2_member``) is served by that
    representative and listed in ``process-groups.json``; every other process
    is generated on its own. A request's explicit ``ProcessAlias`` records
    are published with its process (the manifest's alias records), and such
    a request is never folded into another (as in the tree engine).

    One ``preparation_cache.PreparationMemo`` serves every model
    preparation of the call (the order inference, the process slices, the
    probe and Goldstone slices) and is dropped with it; its persistent store
    of lowered programs (``preparation_cache.store_for``, unless off) serves
    them across generations. Its state is logged, never written into the
    artifact.
    """
    from pyamplicol.artifacts.process_groups import (
        grouping_process,
        initial_process_groups,
    )
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.preparation_cache import PreparationMemo, store_for
    from pyamplicol.loops.v2.artifact import generate_process, write_plan
    from pyamplicol.loops.v2.blocks import born_dag_scope

    from .process_grouping import (
        GroupingBasis,
        GroupMember,
        LegMapping,
        invert_permutation,
    )

    _raise_blocked(backend, processes)
    reporter = GenerationPhaseReporter(backend._progress)
    reporter.start_generation(
        total=3, details={"execution_mode": "loop_recurrence", "engine": "v2"}
    )
    started = time.perf_counter()
    grouping = backend._run_config.process.flavor_blind
    memo = PreparationMemo(store_for(backend._run_config.model))
    stack = ExitStack()
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
            # Their Born DAGs stay in the scope: each slice reuses its own
            # (``blocks.born_dag_scope``; compiled once per process, R3).
            stack.enter_context(born_dag_scope())
            # Only the scope keeps the Born DAGs (each slice releases its own).
            context, requests, _released, skipped = _born_members(
                backend, context, memo
            )
            del _released
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
                        if grouping and not entry.aliases
                        else None
                    )
                    if found is not None:
                        members.setdefault(found[0], []).append((index, found[1]))
                        phase.update(index + 1, message=f"{name} (flavour member)")
                        continue
                    process = generate_process(
                        compiled, request, native, correlators, memo=memo
                    )
                    # The validation points of a loop-induced process (None
                    # for a virtual), for its filter and its probes.
                    candidates = _loop_induced_candidates(
                        context, entry, process, validation, index
                    )
                    # M-LIT lit-fold (USER ~22:39, F1): the fold of a
                    # loop-induced process's certified proportional chains
                    # (None for a virtual or with generation.
                    # loop_induced_folding off); a folded process is
                    # exported again, folded.
                    folding, process = _loop_induced_fold(
                        backend, context, process, request, candidates, correlators
                    )
                    # m-lit Q10 (LIT-R2): the exact helicity-zero filter of a
                    # loop-induced process (None for a virtual); rows it
                    # drops are exported again without their row.
                    helicity, process = _loop_induced_filter(
                        backend, context, process, request, candidates, correlators
                    )
                    # m-lit Q12: the generation-time probes of a loop-induced
                    # process, run only with generation.loop_induced_probes
                    # (its record; None for a virtual).
                    probes = _loop_induced_probes(
                        backend, context, process, native, candidates, memo
                    )
                    generated[name] = process
                    path = Path(temporary) / f"{name}.pacbin"
                    write_plan(
                        process.plan,
                        path,
                        native=process.native,
                        scaleless=process.scaleless_plan,
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
                                # Loop-induced: kind, interference, poles
                                # (m-lit LIT-AMP; absent for a virtual).
                                **process.contribution(),
                                # A TIR-only process's one backend (m-lit
                                # lit-eft; None otherwise, not recorded).
                                reductions=process.reductions(),
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
                                **(
                                    {"loop_induced_probes": probes}
                                    if probes is not None
                                    else {}
                                ),
                                **(
                                    {"loop_induced_helicity_filter": helicity}
                                    if helicity is not None
                                    else {}
                                ),
                                **(
                                    {"loop_induced_fold": folding}
                                    if folding is not None
                                    else {}
                                ),
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
                            aliases=entry.aliases,
                            request=dict(process.request),
                            plan_sha256=process.plan_sha256,
                            scaleless_plan_sha256=process.scaleless_plan_sha256,
                            state_legs=tuple(legs),
                            helicities=tuple(map(tuple, process.helicities())),
                            point_inputs=process.point_inputs,
                        )
                    )
                    phase.update(index + 1, message=name)
            # The preparation memo and store state (log only: the artifact
            # is the same whatever the cache state).
            _LOGGER.info("loop preparation: %s", json.dumps(memo.report()))
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
                                # Representative leg -> member leg.
                                LegMapping(
                                    invert_permutation(member.legs),
                                    False,
                                    "exact-loop-species"
                                    if member.species
                                    else "side-permutation",
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
    finally:
        stack.close()
    reporter.finish_generation()
    return GenerationResult(
        output=result.output,
        processes=ProcessSet(
            tuple(e.request for e in context.expanded),
            # The published aliases (as the tree service returns them).
            tuple(
                alias
                for alias in processes.aliases
                if alias.process_name in {e.request.name for e in context.expanded}
            ),
        ),
        mode=mode,
        files=result.files,
    )


def check_loop_requests(
    backend,
    processes,
    *,
    model=None,
    points: int,
    seed: int,
    precision=None,
    stored=None,
    sqrt_s=None,
    arbitrary_digits=None,
):
    """``pyamplicol check`` (``loops.v2.checks``): every process the request
    serves, regenerated IN MEMORY through this service's own preparation
    (``_prepare`` and ``_born_members``, as ``generate``, with one
    ``preparation_cache.PreparationMemo`` shared by the processes' and the
    probe / Goldstone slices; nothing written), checked at ``points`` seeded
    phase-space points of the generation's validation convention
    (``_validation_momenta`` at seeds ``seed``, ``seed + 1``, ...: decays at
    rest, scatterings in the centre-of-mass frame; ``sqrt_s`` replaces a
    scattering's energy, ``check.sqrt_s``) in the
    carriers of ``induced.probe_carriers(precision)`` (None: binary64, then
    double-double where binary64 fails, is refused or is out of scope, then
    the arbitrary-precision rung for a row double-double leaves undecided,
    at ``arbitrary_digits`` digits, None: derived per row, ``checks.Rung``).
    ``stored`` (an artifact's manifest, ``artifacts.manifest.
    ArtifactManifest``) guards an artifact input: the prepared model must be
    the stored one (its manifest record: name, source kind, content hash,
    restriction) and the expanded processes exactly the stored ones (id and
    expression), else ``ArtifactError`` before anything is generated.
    Returns one ``checks.ProcessCheck`` per served process."""

    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.preparation_cache import PreparationMemo, store_for
    from pyamplicol.loops.v2.blocks import born_dag_scope
    from pyamplicol.loops.v2.checks import check_request
    from pyamplicol.loops.v2.induced import probe_carriers

    _raise_blocked(backend, processes)
    carriers = probe_carriers(precision)
    memo = PreparationMemo(store_for(backend._run_config.model))
    checked = []
    with born_dag_scope():
        context = _prepare(
            backend,
            processes,
            model or backend._configured_model_source(),
            planning=False,
            phase=PhaseHandle("loop-check", backend._progress, len(processes.requests)),
        )
        if stored is not None:
            _require_stored(context, stored)
        context, requests, _dags, _skipped = _born_members(backend, context, memo)
        del _dags
        native = load_loop_native()
        compiled = context.resolved.compiled
        for entry, request in zip(context.expanded, requests, strict=True):
            name = entry.request.name
            momenta = _validation_momenta(
                context,
                entry,
                seed=seed,
                count=points,
                unavailable=f"check momenta unavailable for {name}",
                sqrt_s=sqrt_s,
            )
            checked.append(
                check_request(
                    name,
                    entry.request.expression,
                    compiled,
                    request,
                    native,
                    momenta,
                    carriers,
                    memo=memo,
                    arbitrary=precision is None,
                    digits=arbitrary_digits,
                )
            )
    return checked


def _require_stored(context, stored) -> None:
    """``check_loop_requests``' artifact guard: the prepared model is the
    stored one (``artifact_writer._model_metadata``, the record the writer
    stores) and the expanded processes are the stored ones (id and
    expression); else ``ArtifactError`` naming the difference."""

    from pyamplicol.api.errors import ArtifactError

    from .artifact_writer import _model_metadata

    prepared = _model_metadata(context.resolved.source, context.resolved.compiled)
    keys = ("name", "source_kind", "content_sha256", "restriction")
    differ = [
        f"{key} {stored.model.get(key)!r} stored, {prepared.get(key)!r} prepared"
        for key in keys
        if stored.model.get(key) != prepared.get(key)
    ]
    if differ:
        raise ArtifactError(
            f"{stored.root}: the model check prepared is not the artifact's "
            f"stored model ({'; '.join(differ)}): check regenerates the stored "
            "processes from their own model; give the stored model's source "
            "(model= / --model)"
        )
    expected = {str(p["id"]): str(p["expression"]) for p in stored.processes}
    regenerated = {
        str(entry.request.name): str(entry.request.expression)
        for entry in context.expanded
    }
    if regenerated != expected:
        raise ArtifactError(
            f"{stored.root}: the regenerated processes {regenerated} are not the "
            f"stored ones {expected}"
        )


def _loop_induced_candidates(context, entry, process, validation, index):
    """The generation's validation points of a loop-induced process
    (``generation.validation.samples`` candidates, at least one): the
    candidate points of its helicity filter and its probes; none for an
    exact colour zero, None for a virtual."""

    slice_ = process.slice
    if getattr(slice_, "amplitude", None) is None:
        return None
    if slice_.amplitude.colour_zero:
        return []
    return _validation_momenta(
        context,
        entry,
        seed=validation.seed + index,
        count=max(validation.samples, 1),
        unavailable="loop-induced probe momenta unavailable",
    )


def _validation_momenta(context, entry, *, seed, count, unavailable, sqrt_s=None):
    """``count`` phase-space points of the generation's validation convention
    (``build_process_validation_point`` at seeds ``seed``, ``seed + 1``,
    ...; ``sqrt_s`` replaces a scattering's energy): the four-vectors, or
    ``GenerationError`` "``unavailable``: <why>"."""

    momenta = []
    for sample in range(count):
        record = build_process_validation_point(
            entry.process_ir,
            context.resolved.model,
            process_id=entry.request.name,
            seed=seed + sample,
            sqrt_s=sqrt_s,
        )
        if not record.available:
            raise GenerationError(f"{unavailable}: {record.error}")
        momenta.append(record.four_vectors)
    return momenta


def _loop_induced_fold(backend, context, process, request, candidates, correlators):
    """The F1 fold of a loop-induced process (M-LIT lit-fold, USER ~22:39
    "Yes, and allow exact zeros"; ``fold.fold_slice``): chains whose
    numerators the exact test certifies proportional are dressed once, a
    group whose weights cancel exactly is dropped, a process left with no
    chain and no counterterm ket is an exact zero. Its record (None for a
    virtual and with ``generation.loop_induced_folding`` off: that artifact
    is the unfolded one) and the process, exported again when anything
    folded. The identities are certified at the filter's stated failure
    probability (``generation.helicity_filter_failure_probability``; 0 folds
    nothing). An exact colour zero has no chain to fold."""

    from pyamplicol.loops.v2 import fold
    from pyamplicol.loops.v2.artifact import reexport_process

    generation = backend._run_config.generation
    if candidates is None or not generation.loop_induced_folding:
        return None, process
    slice_ = process.slice
    failure = generation.helicity_filter_failure_probability
    if slice_.amplitude.colour_zero:
        return {"exact_colour_zero": True}, process
    if failure == 0:
        return fold.disabled(failure, "zero failure probability"), process
    folded = fold.fold_slice(slice_, failure=failure)
    if not folded.families:
        # Nothing folds: the plan keeps its bytes.
        return folded.verdict.record(), process
    slice_.fold = folded
    process = reexport_process(process, context.resolved.compiled, request, correlators)
    return folded.verdict.record(), process


def _loop_induced_filter(backend, context, process, request, candidates, correlators):
    """The exact helicity-zero filter of a loop-induced process (m-lit Q10,
    ``helicity_filter.generation_filter``, at the stated failure probability
    ``generation.helicity_filter_failure_probability``): its record (None for
    a virtual) and the process, exported again without the rows it drops
    (none when it would drop every row: m-lit Q13, the verdict is then a
    diagnostic)."""

    from pyamplicol.loops.v2.artifact import reexport_process
    from pyamplicol.loops.v2.helicity_filter import generation_filter

    if candidates is None:
        return None, process
    slice_ = process.slice
    record, dropped = generation_filter(
        slice_,
        candidates,
        failure=backend._run_config.generation.helicity_filter_failure_probability,
    )
    if dropped:
        slice_.zero_rows = dropped
        process = reexport_process(
            process, context.resolved.compiled, request, correlators
        )
    return record, process


def _loop_induced_probes(backend, context, process, native, candidates, memo=None):
    """The record of the generation-time probes of a loop-induced process
    (m-lit Q12; None for a virtual). Off by default (DECISIONS 2026-11-08
    22:34 (3)): the record is ``{"status": "off"}`` (not run, never a pass;
    an exact colour zero keeps its record) and nothing of the probes runs.
    With ``generation.loop_induced_probes`` they run
    (``induced.generation_probes``: binary64, double-double where a binary64
    probe fails or is refused) on the generation's validation points of the
    process (``_loop_induced_candidates``), and a failure refuses the
    process (``loop-induced-probe``) unless
    ``generation.allow_failed_probes``. The written plan never depends on
    them (it is the generated process's, formed before)."""

    if candidates is None:
        return None
    generation = backend._run_config.generation
    if process.slice.amplitude.colour_zero:
        return {"exact_colour_zero": True}
    fold = getattr(process.slice, "fold", None)
    if fold is not None and fold.exact_zero and generation.loop_induced_probes:
        # An exact fold zero (M-LIT lit-fold): no chain and no counterterm
        # ket is left to probe.
        return {"exact_fold_zero": True}
    if not generation.loop_induced_probes:
        return {"status": "off"}
    from pyamplicol.loops.v2.induced import generation_probes

    return generation_probes(
        process.slice,
        context.resolved.compiled,
        native,
        candidates,
        allow_failure=generation.allow_failed_probes,
        memo=memo,
    )


def _validate_v2_artifact(backend, output, context, generated, members):
    """Load every staged process (and every process it serves: flavour
    members and permutation aliases, ``loops.v2.mapping``) through the public
    runtime and compare its native evaluation, inclusive and for every
    declared colour history it serves, with the Python reference executor of
    the generated slice on the validation points (an alias's in its own leg
    order), in binary64; a point whose binary64 pair misses the tolerance is
    re-validated in both carriers against the reference at 32 digits
    (``_revalidate_v2_point``)."""
    from pyamplicol.api import Runtime
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.loops.v2.mapping import inverse, mapped_histories
    from pyamplicol.loops.v2.observables import FlavourMember, MemberObservables

    manifest = load_manifest(output)
    validation = backend._generation_config.validation
    indices = {entry.request.name: i for i, entry in enumerate(context.expanded)}
    # (public id, representative, member, expanded index, alias permutation)
    served = [(name, name, None, indices[name], None) for name in generated]
    served += [
        (context.expanded[index].request.name, representative, member, index, None)
        for representative, admitted in members.items()
        for index, member in admitted
    ]
    served += [
        (
            str(alias["id"]),
            name,
            FlavourMember(
                str(alias["expression"]),
                context.expanded[indices[name]].request.expression,
                inverse(alias["external_permutation"]),
                {},
            ),
            indices[name],
            tuple(alias["external_permutation"]),
        )
        for name in generated
        for alias in context.expanded[indices[name]].aliases
    ]
    for name, representative, member, index, permutation in served:
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
        if permutation is not None:
            # The target's points in the alias's leg order.
            points = [
                [point[permutation.index(leg)] for leg in range(len(point))]
                for point in points
            ]
        observables = generated[representative].observables
        _validate_v2_points(
            name,
            runtime,
            observables if member is None else MemberObservables(observables, member),
            tuple(history.id for history in observables.histories)
            if member is None
            else tuple(mapped_histories(member, observables.histories)),
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
    from pyamplicol.loops.v2.induced import initial_invariant

    points, scales = [], []
    for vectors in _validation_momenta(
        context,
        entry,
        seed=validation.seed + index,
        count=validation.samples,
        unavailable="virtual validation momenta unavailable",
    ):
        point = [[repr(float(x)) for x in vector] for vector in vectors]
        initial = len(entry.process_ir.initial_pdgs)
        scale = initial_invariant(point, initial)
        if scale <= 0:
            raise GenerationError("virtual validation requires a positive invariant")
        points.append(point)
        scales.append(str(Decimal(scale.numerator) / Decimal(scale.denominator)))
    return points, scales
