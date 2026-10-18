# SPDX-License-Identifier: 0BSD
"""Bounded graph-free loop plans in the existing authenticated PACBIN envelope.

This internal bridge does not activate public loop artifacts or SDK execution.
Full-colour source/legacy model-R2 and certified direct gg execution are admitted.
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


def write_loop_pacbin(program, destination):
    """Lower one checked retained control into plain execution data and bytecode."""
    prepared = program.prepared  # Existing scope/strategy checks remain authoritative.
    bare = program.bare
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
    }
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
                "exact_state": state(family.exact_state),
                "mass_state": state(family.mass_state),
                "projector_state": state(family.projector_state),
                "numerators": family.numerators,
                "rational_numerators": family.rational_numerators,
            }
        )
    for item in prepared.currents:
        data["currents"].append(
            {
                "kind": dict(item.metadata)["insertion"].rule.kind,
                "schedule": schedule(item.instructions),
                "scalar": scalar(item),
                "input_columns": dict(item.metadata)["input_columns"],
            }
        )
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
            import _loop_reduction as backend
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
        born_widths = tuple(
            len(data["schedules"][i]["output_slots"]) for i in data["born"]["schedules"]
        )
        loop_widths = tuple(
            len(data["schedules"][i]["output_slots"])
            for i in data["loops"]["schedules"]
        )
        bare.families = []
        for item in data["families"]:
            shape = family_projector_shape(
                len(bare.helicities),
                born_widths,
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
                    kinematics=SimpleNamespace(propagators=item["propagators"]),
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
        model.color_accuracy, model.color_projection, model.source_scale_view = (
            "full",
            None,
            source_scale,
        )
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
