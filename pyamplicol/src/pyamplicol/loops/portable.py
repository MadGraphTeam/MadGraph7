# SPDX-License-Identifier: 0BSD
"""Bounded graph-free loop plans in the existing authenticated PACBIN envelope.

This internal bridge does not activate public loop artifacts or SDK execution.
Source/legacy model-R2 and certified direct gg execution are admitted.
Exact projected model-R2 weights retain their declared colour accuracy.
Producer graphs, symbolic objects and numerical buffers are never persisted.
The shared Rust core owns the single descriptor/member validation boundary;
Symbolica owns its exact bytecode.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    write_pacbin_atomic,
)

from .family_projectors import (
    bind_family_projector,
    family_projector_shape,
    validate_projector_engine,
)

_PLAN_MEMBER = "schedule/loop-runtime-v3.json"
_BARE_EVALUATORS = (
    "model_evaluator",
    "born_projector",
    "postprocess",
    "zero_poles",
    "sum_order_values",
)


def write_loop_pacbin(
    program,
    destination,
    *,
    resolved=False,
    directed=None,
    directed_histories=None,
    spin_directed=(),
    flavor_members=(),
):
    """Lower one checked retained control into plain execution data and bytecode."""
    from .physical_color import color_projection_metadata

    bare = program.bare
    spin_directed = tuple(spin_directed)
    flavor_members = tuple(flavor_members)
    if flavor_members:
        from .flavor_transport import AdmittedFlavorMember

        if any(not isinstance(item, AdmittedFlavorMember) for item in flavor_members):
            raise TypeError("saved flavour members require exact source admission")
        if len({item.member.process_id for item in flavor_members}) != len(
            flavor_members
        ):
            raise ValueError("saved flavour member IDs must be unique")
    if spin_directed and directed is None:
        raise ValueError("saved spin classes require ordinary directed projections")
    if directed_histories is not None and directed is None:
        raise ValueError("directed history declarations require prepared projections")
    if directed is not None:
        from pyamplicol.color.correlator_matrices import ColorCorrelator

        from .prepared_correlations import PreparedDirectedModel

        if not isinstance(directed, PreparedDirectedModel):
            raise TypeError("directed projection requires prepared model instructions")
        directed.require_owner(program)
        declarations = (
            tuple(
                ColorCorrelator(f"history-{index}", bra.emissions, ket.emissions)
                for index, (bra, ket) in enumerate(directed.bare.histories)
            )
            if directed_histories is None
            else tuple(directed_histories)
        )
        if (
            len(declarations) != len(directed.bare.histories)
            or any(not isinstance(item, ColorCorrelator) for item in declarations)
            or len({item.id for item in declarations}) != len(declarations)
            or any(
                item.bra != bra.emissions or item.ket != ket.emissions
                for item, (bra, ket) in zip(
                    declarations, directed.bare.histories, strict=True
                )
            )
        ):
            raise ValueError(
                "directed history IDs/operations differ from prepared columns"
            )
        if (
            resolved
            or directed.bare.spin_layout is not None
            or program.r2_strategy != "model"
        ):
            raise NotImplementedError(
                "saved directed projections currently require inclusive physical "
                "helicities and model R2; provide spin classes separately "
                "through spin_directed; resolved and direct-R2 "
                "transport remain unsupported"
            )
        from .physical_correlations import directed_color_convention

        convention = directed_color_convention(bare, directed.bare.histories)
    if spin_directed:
        classes = []
        for item in spin_directed:
            if not isinstance(item, PreparedDirectedModel):
                raise TypeError("spin projection requires prepared model instructions")
            item.require_owner(program)
            layout = item.bare.spin_layout
            if layout is None or not layout.legs:
                raise ValueError(
                    "saved spin projection requires a nonempty joint class"
                )
            layout.require_owner(bare)
            if item.bare.histories != directed.bare.histories:
                raise ValueError("spin and ordinary directed history columns differ")
            classes.append(layout.legs)
        if len(set(classes)) != len(classes):
            raise ValueError("duplicate saved joint spin class")
    components = None
    if resolved and getattr(bare, "color_accuracy", "full") == "lc":
        from .color_components import prepare_born_color_components

        components = bare.born_color_components
        if components is None:
            components = prepare_born_color_components(bare.born_color_tensors)
            bare.born_color_components = components
            bare._prepared_process = program._prepared_model = None
        components.require_width(len(bare.born_colors))
    prepared = program.prepared  # Existing scope/strategy checks remain authoritative.
    payloads, payload_ids, schedules, schedule_ids = {}, {}, [], {}

    def state(value):
        if value not in payload_ids:
            name = f"evaluators/exact-{len(payload_ids)}.state"
            payload_ids[value] = name
            payloads[name] = value
        return payload_ids[value]

    def schedule(item):
        instructions = item.instructions
        if instructions not in schedule_ids:
            schedule_ids[instructions] = len(schedules)
            schedules.append(
                {
                    "states": [state(value) for value in instructions.states],
                    "input_slots": instructions.input_slots,
                    "parameter_count": instructions.parameter_count,
                    "output_slots": instructions.output_slots,
                }
            )
        return schedule_ids[instructions]

    def group(executors, columns, sharing):
        # Lower the already proven execution groups, not dormant per-graph maps.
        # Every alias points to the exact representative program and ordered input
        # columns actually executed by _evaluate_coefficient_groups.
        return {
            "schedules": [
                schedule(executors[r].prepared) for r in sharing.representative_for
            ],
            "input_columns": [columns[r] for r in sharing.representative_for],
            "representatives": sharing.representatives,
            "representative_for": sharing.representative_for,
        }

    source = prepared.bare.source_scalars
    binding = {"kind": "legacy"}
    if source is not None:
        owner = asdict(source)
        for name in ("values_state", "defaults_state", "guards_state"):
            owner[name] = state(owner[name]) if owner[name] is not None else None

        def scale(record):
            if record is None:
                return None
            return {
                "outputs": record.outputs,
                "scale_parameter": record.scale_parameter,
                "scale_index": record.scale_index,
                "scale_state": state(record.scale_state),
                "values_state": state(record.values_state),
            }

        binding = {
            "kind": "source",
            "owner": owner,
            "ordinary_scale": scale(prepared.bare.source_scale),
            "counterterm_scale": scale(prepared.source_scale),
            "model_output_indices": bare.model_output_indices,
            "initial_overrides": dict(bare.model_state.snapshot().overrides),
        }

    def scalar(item, *, field=False):
        if source is None:
            return {"kind": "evaluator", "state": state(item.coefficient_state)}
        metadata = dict(item.metadata)
        columns = (
            [metadata["source_scalar_column"]]
            if field
            else metadata["source_scalar_columns"]
        )
        return {"kind": "source_columns", "columns": columns}

    def dual_weights(value, selections):
        rectangle = value.runtime_rectangle(bare)
        requested = value.requested_contraction
        reason = value.direct_reason
        if value is not bare.weight_plan and rectangle is None:
            requested = bare.weight_plan.requested_contraction
            reason = (
                reason or "this observable has no certified rectangular colour bank"
            )
        return {
            "kind": "exact-dual-born-v1",
            "state": state(value.state),
            "columns": value.columns,
            "term_graphs": [
                index
                for index, terms in enumerate(bare.loop_reduction_powers)
                for _ in terms
            ],
            "family_columns": selections,
            "rectangular": None if rectangle is None else rectangle[0],
            "constants_state": None if rectangle is None else state(rectangle[1]),
            "requested_contraction": requested,
            "direct_reason": reason,
        }

    data = {
        "schema": 3,
        "r2_strategy": program.r2_strategy,
        "dimensional_numerators": bare.dimensional_numerators,
        "external_pdgs": [
            edge.particle.pdg_code
            for edge in sorted(
                bare.process.born[0].native.external_edges,
                key=lambda edge: edge.external_index,
            )
        ],
        "state_legs": [asdict(leg) for leg in bare.state_layout.legs],
        "helicities": bare.helicities,
        "initial_average": int(str(bare.initial_average)),
        "model_inputs": [value.get_name() for value in bare.model_inputs],
        "model_count": len(bare.model_names),
        "model_binding": binding,
        "bound_positions": dict(bare.bound_positions),
        "order_layout": asdict(bare.order_layout),
        "schedules": schedules,
        "born": group(bare.born_executors, bare.born_input_columns, bare.born_sharing),
        "loops": group(bare.loop_executors, bare.loop_input_columns, bare.loop_sharing),
        "families": [],
        "bare_evaluators": {
            name: state(bare._evaluator_states[name])
            for name in _BARE_EVALUATORS
            if source is None or name != "model_evaluator"
        },
        "currents": [],
        "fields": [scalar(item, field=True) for item in prepared.fields],
        "model_evaluators": {
            name: state(value) for name, value in prepared.evaluator_states
        },
        "dual_born_weights": dual_weights(
            bare.weight_plan,
            [dict(f.metadata)["weight_columns"] for f in prepared.bare.families],
        ),
    }
    if flavor_members:
        data["admitted_flavor_members"] = [
            item.for_program(program, spin_directed=spin_directed)
            for item in flavor_members
        ]
    projection = color_projection_metadata(bare)
    if projection is not None:
        data["color_projection"] = projection
    for family in prepared.bare.families:
        metadata = dict(family.metadata)
        data["families"].append(
            {
                name: metadata[name]
                for name in (
                    "index",
                    "graph_indices",
                    "sector_indices",
                    "powers",
                    "mass_columns",
                )
            }
            | {
                "propagators": metadata["kinematics"].propagators,
                "kinematics": {
                    "context": metadata["kinematics"].kinematic_context,
                    "mass_output_count": len(metadata["kinematics"].mass_expressions),
                    "on_shell": metadata["kinematics"].on_shell_bindings,
                },
                "exact_state": state(family.exact_state),
                "mass_state": state(family.mass_state),
                "projector_state": state(family.projector_state),
                "numerators": family.numerators,
                "rational_numerators": family.rational_numerators,
            }
        )
    if resolved:
        if program.r2_strategy != "model":
            raise ValueError("resolved projection requires model R2")
        # Only local projection changes: currents are still executed once for
        # every point/helicity in a single native batch. LC adds signed Born
        # anchors after that shared work; inclusive and NLC/full keep contraction.
        bare._make_born_projector(
            (bare.born_metric,)
            if components is None
            else components.mask_bra(bare.born_metric),
            helicities=bare.helicities[:1],
            real_result=True,
            _record_name="helicity_born_projector",
        )
        family_projectors = []
        from .dual_weights import select_dual_born_weights

        resolved_weights = select_dual_born_weights(
            bare,
            bare.weight_plan.requested_contraction,
            resolved=components is not None,
        )
        selections = []
        for family in prepared.bare.families:
            metadata = dict(family.metadata)
            name = f"helicity_family_projector_{metadata['index']}"
            bare._compile_weighted_family_projection(
                metadata["graph_indices"],
                metadata["powers"],
                metadata["sector_indices"],
                weights=resolved_weights,
                helicities=bare.helicities[:1],
                _record_name=name,
            )
            family_projectors.append(state(bare._evaluator_states[name]))
            selections.append(
                resolved_weights.family_columns(
                    bare, metadata["graph_indices"], metadata["sector_indices"]
                )
            )
        program._make_projector(
            bare.helicities[:1],
            color_components=components,
            _record_name="helicity_projector",
        )
        data["helicity_projection"] = {
            "born": state(bare._evaluator_states["helicity_born_projector"]),
            "families": family_projectors,
            "counterterms": state(program._evaluator_states["helicity_projector"]),
            "dual_born_weights": dual_weights(resolved_weights, selections),
        }
        if components is not None:
            data["born_color_components"] = components.as_dict()
    for item in prepared.currents:
        data["currents"].append(
            {
                "kind": dict(item.metadata)["insertion"].rule.kind,
                "schedule": schedule(item.instructions),
                "scalar": scalar(item),
                "input_columns": dict(item.metadata)["input_columns"],
            }
        )
    if directed is not None:

        def directed_data(value):
            return {
                "kind": "directed-complex",
                "convention": convention,
                "color_legs": [
                    (leg.label, leg.representation) for leg in value.bare.color_legs
                ],
                "histories": [item.to_json_dict() for item in declarations],
                "born": state(value.bare.born_state),
                "families": [state(family.state) for family in value.bare.families],
                "postprocess": state(value.bare.postprocess_state),
                "counterterms": state(value.counterterm_state),
                "assemble": state(value.assembly_state),
                "sum_order_values": state(value.sum_order_state),
                "dual_born_weights": dual_weights(
                    value.bare.weight_plan,
                    [f.weight_columns for f in value.bare.families],
                ),
            }

        data["directed_projection"] = directed_data(directed)
        if spin_directed:
            data["spin_projections"] = [
                {
                    "legs": item.bare.spin_layout.legs,
                    "helicities": item.bare.helicities,
                    "phases": [
                        state(phase) for phase in item.bare.spin_layout.phase_states
                    ],
                    "source_contracts": [
                        source.to_json_dict()
                        for source in item.bare.spin_layout.contracts
                    ],
                    "directed": directed_data(item),
                }
                for item in spin_directed
            ]
    members = [
        PacbinMemberSource(
            _PLAN_MEMBER,
            PacbinMemberKind.LOOP_RUNTIME_PLAN,
            io.BytesIO(json.dumps(data, separators=(",", ":")).encode()),
        )
    ]
    members.extend(
        PacbinMemberSource(
            name, PacbinMemberKind.SYMBOLICA_EXACT_STATE, io.BytesIO(value)
        )
        for name, value in payloads.items()
    )
    return write_pacbin_atomic(destination, members)


def _unsupported(*args, **kwargs):
    raise NotImplementedError(
        "portable control supports ordinary complete evaluation and arithmetic "
        "rebinding; producer-dependent modification, correlations and rescue "
        "are unavailable"
    )


@dataclass(frozen=True, slots=True)
class PortableLoopPlan:
    """Native-owned validated data; every bind creates fresh arithmetic workspaces."""

    _native: object
    _backend: object

    @classmethod
    def open(cls, path, *, backend=None):
        if backend is None:
            from .native import load_loop_native

            backend = load_loop_native()
        return cls(backend.PortableLoopPlan(str(path)), backend)

    def bind(
        self,
        *,
        arithmetic="double",
        precision=None,
        reduction_engine="generic",
        coefficient_engine="interpreted",
        projector_engine="interpreted",
        parameters=None,
    ):
        from .kernels import CoefficientSharingGroups
        from .orders import BornLoopPair, VirtualOrderSector
        from .physical_counterterms import NativeModelVirtualProgram
        from .physical_orders import BornSquaredSector, PhysicalOrderLayout
        from .physical_process import NativeBareProcessProgram
        from .physical_states import ExternalStateLayout, ExternalStateLeg

        data, backend = json.loads(self._native.metadata_json()), self._backend
        if data.get("directed_projection") is not None:
            raise NotImplementedError(
                "saved directed-complex projections require the native directed "
                "scheduler; ordinary evaluation cannot execute this artifact"
            )
        options = dict(arithmetic=arithmetic, precision=precision)
        state = self._native.evaluator_state
        binding = data["model_binding"]
        source = binding["kind"] == "source"
        if parameters is not None and not source:
            raise ValueError("named parameters require a source-owned portable plan")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("external updates must be a parameter mapping")

        class Bare(NativeBareProcessProgram):
            set_color_metrics = _unsupported
            evaluate_ordered_partial_native = _unsupported
            rebind = _unsupported
            prepared = property(_unsupported)

            def __getattr__(self, name):
                if name in {
                    "process",
                    "recurrences",
                    "loop_graphs",
                    "born_dags",
                    "model_expressions",
                    "born_color_tensors",
                    "loop_color_tensors",
                }:
                    _unsupported()
                raise AttributeError(name)

        class Model(NativeModelVirtualProgram):
            evaluate_ordered_partial_native = _unsupported
            prepared = property(_unsupported)

            def evaluate_ordered_native(
                self, momenta, scales_squared, *, model_values=None, point_ids=None
            ):
                currents = self.bare.prepare_currents_native(
                    momenta,
                    scales_squared,
                    model_values=model_values,
                    point_ids=point_ids,
                )
                return self._complete_native(
                    self.bare._evaluate_currents_native(currents)
                )

            def rebind(
                self,
                *,
                arithmetic,
                precision=None,
                reduction_engine="generic",
                coefficient_engine="interpreted",
                projector_engine="interpreted",
                parameters=None,
            ):
                if self.bare.model_state is not None:
                    if parameters is not None and not isinstance(parameters, Mapping):
                        raise TypeError("external updates must be a parameter mapping")
                    parameters = dict(self.bare.model_state.snapshot().overrides) | (
                        {} if parameters is None else dict(parameters)
                    )
                return self._portable_plan.bind(
                    arithmetic=arithmetic,
                    precision=precision,
                    reduction_engine=reduction_engine,
                    coefficient_engine=coefficient_engine,
                    projector_engine=projector_engine,
                    parameters=parameters,
                )

            def update_parameters(self, changes):
                if self.bare.model_state is None:
                    raise ValueError(
                        "named parameters require a source-owned portable plan"
                    )
                return self.bare.model_state.update(changes)

        bare, model = Bare.__new__(Bare), Model.__new__(Model)
        bare.backend, bare.options = backend, options
        bare.arithmetic, bare.precision = arithmetic, precision
        bare.projector_engine = validate_projector_engine(projector_engine, arithmetic)
        bare.reduction_engine, bare.coefficient_engine = (
            reduction_engine,
            coefficient_engine,
        )
        bare.state_layout = ExternalStateLayout(
            tuple(ExternalStateLeg(**leg) for leg in data["state_legs"])
        )
        bare.state_kinds = tuple(leg.kind for leg in bare.state_layout.legs)
        bare.helicities = tuple(map(tuple, data["helicities"]))
        bare.initial_average = data["initial_average"]
        bare.initial_spin_average = bare.state_layout.initial_spin_average
        bare.model_inputs = tuple(data["model_inputs"])
        bare.bound_positions = data["bound_positions"]
        bare.model_state = bare.model_scale_view = None
        bare.dimensional_numerators = data["dimensional_numerators"]
        source_scale = None
        if source:
            from .scalar_bindings import PreparedScalarBindings, PreparedScaleBindings

            item = binding["owner"]
            owner = PreparedScalarBindings(
                tuple(item["external_names"]),
                tuple(item["external_types"]),
                tuple(map(tuple, item["outputs"])),
                tuple((name, tuple(value)) for name, value in item["fixed_parameters"]),
                state(item["values_state"]),
                state(item["defaults_state"]) if item["defaults_state"] else None,
                state(item["guards_state"]) if item["guards_state"] else None,
            )
            bare.model_state = owner.bind(
                backend=backend,
                parameters=dict(binding["initial_overrides"])
                | ({} if parameters is None else dict(parameters)),
                **options,
            )
            bare.model_evaluator = None
            bare.model_output_indices = tuple(binding["model_output_indices"])

            def scale_view(item):
                if item is None:
                    return None
                return PreparedScaleBindings(
                    owner,
                    tuple(map(tuple, item["outputs"])),
                    item["scale_parameter"],
                    item["scale_index"],
                    state(item["scale_state"]),
                    state(item["values_state"]),
                ).bind(bare.model_state)

            bare.model_scale_view = scale_view(binding["ordinary_scale"])
            source_scale = scale_view(binding["counterterm_scale"])

        def order(value):
            return tuple((name, power) for name, power in value)

        layout = data["order_layout"]
        bare.order_layout = PhysicalOrderLayout(
            tuple(
                BornSquaredSector(
                    order(s["order"]),
                    tuple((order(a), order(b)) for a, b in s["pairs"]),
                )
                for s in layout["born"]
            ),
            tuple(
                VirtualOrderSector(
                    order(s["order"]),
                    tuple(
                        BornLoopPair(order(p["born"]), order(p["loop"]))
                        for p in s["pairs"]
                    ),
                )
                for s in layout["virtual"]
            ),
            tuple(layout["correction_names"]),
        )
        # Repeated references share one schedule only within this fresh view.
        plans = [
            backend.CoefficientPlan(
                [state(name) for name in item["states"]],
                item["input_slots"],
                item["parameter_count"],
                item["output_slots"],
                engine=coefficient_engine,
                **options,
            )
            for item in data["schedules"]
        ]
        bare.coefficient_engine = plans[data["born"]["schedules"][0]].engine
        for name in ("born", "loop"):
            group = data["born" if name == "born" else "loops"]
            setattr(
                bare,
                name + "_executors",
                tuple(plans[index] for index in group["schedules"]),
            )
            setattr(
                bare, name + "_input_columns", tuple(map(tuple, group["input_columns"]))
            )
            setattr(
                bare,
                name + "_sharing",
                CoefficientSharingGroups(
                    tuple(group["representatives"]), tuple(group["representative_for"])
                ),
            )

        def evaluator(name):
            return backend.CoefficientEvaluator(state(name), **options)

        for name, reference in data["bare_evaluators"].items():
            setattr(bare, name, evaluator(reference))
        weights = data.get("dual_born_weights")
        if weights is not None and weights.get("rectangular") is not None:
            bare.dual_born_weights = backend.DualBornWeights(
                state(weights["state"]),
                json.dumps(weights["rectangular"]),
                state(weights["constants_state"]),
                weights["columns"],
                **options,
            )
        else:
            bare.dual_born_weights = (
                None if weights is None else evaluator(weights["state"])
            )
        born_widths = tuple(
            len(data["schedules"][i]["output_slots"]) for i in data["born"]["schedules"]
        )
        loop_widths = tuple(
            len(data["schedules"][i]["output_slots"])
            for i in data["loops"]["schedules"]
        )
        bare.families = []
        for family_index, item in enumerate(data["families"]):
            columns = (
                ()
                if weights is None
                else tuple(weights["family_columns"][family_index])
            )
            shape = family_projector_shape(
                len(bare.helicities),
                born_widths if weights is None else (len(columns),),
                loop_widths,
                item["graph_indices"],
                len(item["sector_indices"]),
                len(item["powers"]),
            )
            bare.families.append(
                SimpleNamespace(
                    index=item["index"],
                    graph_indices=tuple(item["graph_indices"]),
                    sector_indices=tuple(item["sector_indices"]),
                    powers=tuple(map(tuple, item["powers"])),
                    weight_columns=columns,
                    kinematics=SimpleNamespace(
                        propagators=item["propagators"],
                        on_shell_bindings=tuple(
                            map(
                                tuple,
                                (item.get("kinematics") or {}).get("on_shell", []),
                            )
                        ),
                    ),
                    exact_evaluator=backend.ExactPolynomialEvaluator(
                        state(item["exact_state"])
                    ),
                    masses=evaluator(item["mass_state"]),
                    mass_columns=tuple(item["mass_columns"]),
                    projector=bind_family_projector(
                        state(item["projector_state"]),
                        *shape,
                        backend=backend,
                        options=options,
                        engine=projector_engine,
                    ),
                    reducer=backend.PolynomialPlan(
                        item["propagators"],
                        item["numerators"],
                        [f"c{i}" for i in range(len(item["powers"]))],
                        engine=reduction_engine,
                        **options,
                    ),
                    rational_reducer=(
                        backend.PolynomialPlan(
                            item["propagators"],
                            item["rational_numerators"],
                            [f"c{i}" for i in range(len(item["powers"]))],
                            engine=reduction_engine,
                            **options,
                        )
                        if item["rational_numerators"] != ["0", "0", "0"]
                        else None
                    ),
                )
            )
        bare.order_channels = tuple(
            (sector, f.index) for f in bare.families for sector in f.sector_indices
        )
        model.bare, model.r2_strategy = bare, data["r2_strategy"]
        from .physical_color import restore_color_projection

        model.color_projection = restore_color_projection(
            bare, data.get("color_projection")
        )
        model.color_accuracy = bare.color_accuracy
        from .color_components import BornColorComponents

        components = data.get("born_color_components")
        bare.born_color_components = (
            None
            if components is None
            else BornColorComponents(
                tuple(components["ids"]), tuple(map(tuple, components["members"]))
            )
        )
        model.source_scale_view = source_scale
        model.currents = [
            SimpleNamespace(
                coefficient_evaluator=(
                    None if source else evaluator(item["scalar"]["state"])
                ),
                source_scalar_columns=(
                    tuple(item["scalar"]["columns"]) if source else ()
                ),
                executor=plans[item["schedule"]],
                input_columns=tuple(item["input_columns"]),
            )
            for item in data["currents"]
        ]
        model.field_currents = [
            SimpleNamespace(
                coefficient_evaluator=None if source else evaluator(item["state"]),
                source_scalar_column=item["columns"][0] if source else None,
            )
            for item in data["fields"]
        ]
        for name, reference in data["model_evaluators"].items():
            setattr(model, name, evaluator(reference))
        model._portable_plan = self
        return model
