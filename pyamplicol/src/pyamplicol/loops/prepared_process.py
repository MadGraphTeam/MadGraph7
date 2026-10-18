# SPDX-License-Identifier: 0BSD
"""Retained internal cold plans, separate from mutable arithmetic workspaces.

Only exact instructions and binding metadata are retained here. Producer graph
references support existing diagnostics; this is not an artifact format, and no
Python graph or native evaluator is serialized by these records.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from .kernels import NativeCoefficientDAG, PreparedCoefficientDAG


def _metadata(owner, names):
    """Freeze container layouts; the native symbolic definitions are immutable."""

    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType(dict(value))
        if isinstance(value, list):
            return tuple(value)
        return value

    return tuple((name, freeze(getattr(owner, name))) for name in names)


@dataclass(frozen=True, slots=True)
class PreparedFamily:
    metadata: tuple[tuple[str, Any], ...]
    exact_state: bytes
    mass_state: bytes
    projector_state: bytes
    numerators: tuple[str, str, str]
    rational_numerators: tuple[str, str, str]

    @classmethod
    def from_family(cls, family, evaluator_states):
        from .physical_process import _reduction_numerators

        names = [f"c{i}" for i in range(len(family.powers))]
        return cls(
            _metadata(
                family,
                (
                    "index",
                    "graph_indices",
                    "sector_indices",
                    "powers",
                    "kinematics",
                    "mass_parameters",
                    "mass_columns",
                ),
            ),
            family.exact_state,
            evaluator_states[f"family{family.index}_masses"],
            evaluator_states[f"family{family.index}_projector"],
            tuple(_reduction_numerators(family.powers, names)),
            tuple(_reduction_numerators(family.powers, names, rational=True)),
        )

    def bind(self, backend, options, reduction_engine):
        from .physical_process import _FamilyProgram

        metadata = dict(self.metadata)
        parameters = [f"c{i}" for i in range(len(metadata["powers"]))]

        def reducer(sectors):
            return backend.PolynomialPlan(
                metadata["kinematics"].propagators,
                sectors,
                parameters,
                engine=reduction_engine,
                **options,
            )

        return _FamilyProgram(
            **metadata,
            exact_evaluator=backend.ExactPolynomialEvaluator(self.exact_state),
            masses=backend.CoefficientEvaluator(self.mass_state, **options),
            projector=backend.CoefficientEvaluator(self.projector_state, **options),
            reducer=reducer(self.numerators),
            rational_reducer=(
                reducer(self.rational_numerators)
                if self.rational_numerators != ("0", "0", "0")
                else None
            ),
            exact_state=self.exact_state,
        )


_BARE_METADATA = (
    "order_layout",
    "vertex_spinor_frame",
    "model_expressions",
    "loop_graphs",
    "process",
    "external_masses",
    "state_layout",
    "initial_spin_average",
    "initial_average",
    "state_kinds",
    "helicities",
    "dimensional_numerators",
    "cache_coefficient_kernels",
    "share_coefficient_schedules",
    "model_inputs",
    "scale_squared_symbol",
    "model_names",
    "model_positions",
    "model_output_indices",
    "recurrences",
    "momenta",
    "momentum_parameters",
    "external_states",
    "bound_positions",
    "bound_columns",
    "born_recurrences",
    "born_dags",
    "born_term_graphs",
    "born_color_tensors",
    "born_colors",
    "loop_finals",
    "loop_term_offsets",
    "loop_color_tensors",
    "born_term_orders",
    "born_sharing",
    "loop_sharing",
    "born_input_columns",
    "loop_input_columns",
    "loop_reduction_powers",
    "born_metric",
    "loop_metric",
    "order_channels",
)


@dataclass(frozen=True, slots=True)
class PreparedBareProcess:
    metadata: tuple[tuple[str, Any], ...]
    born: tuple[PreparedCoefficientDAG, ...]
    loops: tuple[PreparedCoefficientDAG, ...]
    families: tuple[PreparedFamily, ...]
    evaluator_states: tuple[tuple[str, bytes], ...]
    color_metrics_revision: int

    @classmethod
    def from_program(cls, program):
        if (
            program.model_state is not None
            or getattr(program, "_formal_color_projected", False)
            or getattr(program, "color_accuracy", "full") != "full"
            or getattr(program, "born_color_degree", None) is not None
            or getattr(program, "virtual_color_degree", None) is not None
        ):
            raise NotImplementedError(
                "retained process views currently require unprojected "
                "legacy model bindings"
            )
        return cls(
            _metadata(program, _BARE_METADATA),
            tuple(executor.prepared for executor in program.born_executors),
            tuple(executor.prepared for executor in program.loop_executors),
            tuple(
                PreparedFamily.from_family(family, program._evaluator_states)
                for family in program.families
            ),
            tuple(program._evaluator_states.items()),
            getattr(program, "_color_metrics_revision", 0),
        )

    def bind(
        self,
        *,
        backend,
        arithmetic,
        precision=None,
        reduction_engine="generic",
        coefficient_engine="interpreted",
    ):
        from .color_bilinears import PhysicalColorSpace
        from .physical_process import NativeBareProcessProgram

        view = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
        for name, value in self.metadata:
            setattr(view, name, value)
        view.backend = backend
        view.arithmetic, view.precision = arithmetic, precision
        view.options = dict(arithmetic=arithmetic, precision=precision)
        view.reduction_engine = reduction_engine
        view.coefficient_kernel_cache = (
            backend.CoefficientKernelCache(engine=coefficient_engine, **view.options)
            if view.cache_coefficient_kernels
            else None
        )

        def bind(schedule):
            return NativeCoefficientDAG(
                schedule,
                backend=backend,
                coefficient_engine=coefficient_engine,
                kernel_cache=view.coefficient_kernel_cache,
                **view.options,
            )

        view.born_executors = tuple(map(bind, self.born))
        view.loop_executors = tuple(map(bind, self.loops))
        view.coefficient_engine = view.born_executors[0].coefficient_engine
        view.families = [
            f.bind(backend, view.options, reduction_engine) for f in self.families
        ]
        view._evaluator_states = dict(self.evaluator_states)
        for name in (
            "model_evaluator",
            "born_projector",
            "postprocess",
            "zero_poles",
            "sum_order_values",
        ):
            setattr(
                view,
                name,
                backend.CoefficientEvaluator(
                    view._evaluator_states[name], **view.options
                ),
            )
        view.model_state = view.model_scale_view = None
        view.color_space = PhysicalColorSpace()
        view._color_metrics_revision = self.color_metrics_revision
        view._prepared_process = self
        return view


@dataclass(frozen=True, slots=True)
class PreparedCountertermCurrent:
    metadata: tuple[tuple[str, Any], ...]
    instructions: PreparedCoefficientDAG | None
    coefficient_state: bytes

    @classmethod
    def from_current(cls, current):
        return cls(
            _metadata(
                current,
                tuple(
                    field.name
                    for field in fields(current)
                    if field.name
                    not in {"executor", "coefficient_evaluator", "coefficient_state"}
                ),
            ),
            current.executor.prepared if hasattr(current, "executor") else None,
            current.coefficient_state,
        )

    def bind(self, bare, executors):
        from .physical_counterterms import _CountertermCurrent, _ExternalFieldCurrent

        metadata = dict(self.metadata)
        metadata["coefficient_state"] = self.coefficient_state
        metadata["coefficient_evaluator"] = bare.backend.CoefficientEvaluator(
            self.coefficient_state, **bare.options
        )
        if self.instructions is None:
            return _ExternalFieldCurrent(**metadata)
        key = id(self.instructions)
        if key not in executors:
            executors[key] = NativeCoefficientDAG(
                self.instructions,
                backend=bare.backend,
                coefficient_engine=bare.coefficient_engine,
                kernel_cache=bare.coefficient_kernel_cache,
                **bare.options,
            )
        return _CountertermCurrent(**metadata, executor=executors[key])


@dataclass(frozen=True, slots=True)
class PreparedModelProcess:
    bare: PreparedBareProcess
    metadata: tuple[tuple[str, Any], ...]
    currents: tuple[PreparedCountertermCurrent, ...]
    fields: tuple[PreparedCountertermCurrent, ...]
    evaluator_states: tuple[tuple[str, bytes], ...]

    @classmethod
    def from_program(cls, program):
        if (
            program.r2_strategy not in {"model", "direct"}
            or program.color_projection is not None
            or program.source_scale_view is not None
        ):
            raise NotImplementedError(
                "retained model views currently require full-colour "
                "model or direct-R2 legacy bindings"
            )
        if program.r2_strategy == "direct":
            if not program.bare.dimensional_numerators:
                raise ValueError("retained direct R2 requires dimensional numerators")
            if any(item.rule.kind == "R2" for item in program.insertions) or any(
                current.insertion.rule.kind == "R2" for current in program.currents
            ):
                raise ValueError("retained direct R2 cannot include model R2 currents")
        return cls(
            program.bare.prepared,
            _metadata(
                program,
                (
                    "r2_strategy",
                    "color_accuracy",
                    "color_projection",
                    "source_bindings",
                    "sector_pairs",
                    "matched_insertions",
                    "insertions",
                    "external_fields",
                ),
            ),
            tuple(map(PreparedCountertermCurrent.from_current, program.currents)),
            tuple(map(PreparedCountertermCurrent.from_current, program.field_currents)),
            tuple(program._evaluator_states.items()),
        )

    def bind(self, *, backend, **options):
        from .physical_counterterms import NativeModelVirtualProgram

        view = NativeModelVirtualProgram.__new__(NativeModelVirtualProgram)
        view.bare = self.bare.bind(backend=backend, **options)
        for name, value in self.metadata:
            setattr(view, name, value)
        executors = {}
        view.currents = [
            current.bind(view.bare, executors) for current in self.currents
        ]
        view.field_currents = [
            current.bind(view.bare, executors) for current in self.fields
        ]
        view._evaluator_states = dict(self.evaluator_states)
        for name, state in self.evaluator_states:
            setattr(
                view, name, backend.CoefficientEvaluator(state, **view.bare.options)
            )
        view.source_scale_view = None
        view.bare._model_color_metrics_bound = True
        view.bare._formal_color_projected = False
        view._prepared_model = self
        return view
